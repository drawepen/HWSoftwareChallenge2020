#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h>///////////linux
#include<string.h>
#include<sys/stat.h>
#include<fcntl.h>
#include "set"
#include <sys/mman.h>
#include <sys/time.h>///////////////////
#include<sys/wait.h>
#include <arm_neon.h>


using namespace std;
/**
 * -test文件应该没有符号，格式统一、、
 * -train文件有未归一化数据，存在-号和>1
 * -去除含-号的数据/////不理想
 * -多进程-16
 * -线程创建的进程执行更多行
 * -只使用一位小数
 * -特征分级为1,2,4,8，以便只用移位运算//////失败
 * -特征权值紧挨,连续特征权值//效果不大
 * -训练也开启多进程
 * -以7*recNum估计一行大小//改为以recNum*49/8估计大小，实验表明训练数据越连续预测效果越好，原因不详，大概是训练集没有打乱
 * -使用arm_neon
 * -设置cpu亲和性
 */

int FileSize(const char* fname)
{
    struct stat statbuf;//获取的文件大小含索引，不能用
    if(stat(fname,&statbuf)==0)
        return statbuf.st_size;
    return -1;
}
//设置CPU亲和性
void setCpu(int cid){

    cpu_set_t mask;
    CPU_ZERO(&mask);    /* 初始化set集，将set置为空*/
    CPU_SET(cid, &mask);  /* 依次将0号cpu加入到集合*/
    /*设置cpu 亲和性（affinity）*/
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("Set CPU affinity failue, ERROR:%s\n", strerror(errno));
    }

//    usleep(100);
}
//加载训练数据并训练
int loadTrain(const char* file, float tfs[]) {
    int recNum=1000;
    int totalLn=7600+1;//读数据行数//8000
    int proN=6;//进程数
    char *datas;
    int tjd=0;

    int fd=open(file,O_RDONLY);//O_RDONLY只读
    if(fd==-1){
        printf("%s\n",strerror(errno));
    }else{
        void *addr2=mmap(NULL,totalLn*recNum*7, PROT_READ, MAP_PRIVATE, fd, 0); //mmap映射系统内存池到进程内存
        if (MAP_FAILED == addr2) {
            printf("mmap: %s\n", strerror(errno));
        }
        datas=(char *)addr2;
    }
    //确定维度
    recNum=0;
    while(datas[tjd]!='\n'){
        if(datas[tjd]==','){
            recNum+=1;
        }
        tjd+=1;
    }
    //多进程
    /*创建共享内存*/
    void *addr=mmap(NULL,(proN)*(recNum+1)*2*sizeof(int)+1,PROT_READ | PROT_WRITE,MAP_SHARED | MAP_ANON,-1,0);
    if (MAP_FAILED == addr) {
        printf("mmap: %s\n", strerror(errno));
    }
    char *shm=(char *)addr;
    /**/
    /*分配进程相应任务量,以行为单位,4x,3x...2x...2x*/
    pid_t proH[proN];//记进程proId与进程号的对应
    int proId=0;
    int proAddR[proN];//起始行
    int bc=500;//*16
    proAddR[0]=1;
    proAddR[1]=bc*4+proAddR[0];
    proAddR[2]=bc*2+proAddR[1];
    proAddR[3]=bc*2+proAddR[2];
    proAddR[4]=bc*2+proAddR[3];
    proAddR[5]=bc*1+proAddR[4];
    /*开启进程*/

    int readln=1000;//读数据行数
    int readMaxJd=totalLn*recNum*49/8;//最大读入//防止重复读//不×7，尽量连续读
    for(int i=0;i<proN-1;++i){
        pid_t fpid=fork();//进程号
        if (fpid < 0)
            printf("error in fork!");
        else if (fpid == 0) {//子进程
            break;
        }else {//父进程
            proH[proId]=fpid;
        }
        proId+=1;
    }
//    struct timeval tv0;/////////////////////
//    gettimeofday(&tv0, NULL);/////////////////
//    cout<<tv0.tv_usec<<"进程t"<<proId<<"开始"<<endl;
    /*读数据*/
    if(proId==(proN-1)){
        readln=totalLn-proAddR[proId];
    }else{
        readln=proAddR[proId+1]-proAddR[proId];
        readMaxJd=proAddR[proId+1]*recNum*49/8;
    }
//    cout<<"进程t"<<proId<<"读行"<<readln<<endl;/////////////////


    int *tempf=new int[readln*recNum];
    int labes[readln];
    memset(labes,0,readln*sizeof(int));
    int jf=-1,jln=0,jnow=0,jd=proAddR[proId]*recNum*49/8;
    //确定jd
    while(datas[jd-1]!='\n'){
        ++jd;
    }
//    cout<<"进程t"<<proId<<"jd="<<jd<<endl;/////////////////

    /*
     * 格式必须统一才可读取
     */
    int tfsI[2*recNum+2];
    memset(tfsI,0,2*(recNum+1)*sizeof(int));//大小指的是字节大小
    while(true){
        if(datas[jd]=='-'){
            tempf[++jf]='0'-datas[jd+3];//字符数组转浮点数
            jd+=7;
        }else{
            tempf[++jf]= datas[jd+2]-'0';//字符数组转浮点数
            jd += 6;
        }
        if(++jnow>=recNum){
            if(datas[jd]-'0'){
                labes[jln]=0x0fffffff;
            }
            jln+=1;
            if(jln>=readln){
                break;
            }
            jd+=2;
            jnow=0;
            if(jd>=readMaxJd){
                readln=jln;
            }
        }
    }
    int sl=(recNum)*readln;
    int ji=0,jtf=0,jln2=0;
    int tadd=recNum&labes[jln2];
    int count01[2]={0,0};
    while(ji<sl){
        tfsI[tadd+jtf]  +=tempf[ji];
        tfsI[tadd+jtf+1]+=tempf[ji+1];
        tfsI[tadd+jtf+2]+=tempf[ji+2];
        tfsI[tadd+jtf+3]+=tempf[ji+3];
        tfsI[tadd+jtf+4]+=tempf[ji+4];
        tfsI[tadd+jtf+5]+=tempf[ji+5];
        tfsI[tadd+jtf+6]+=tempf[ji+6];
        tfsI[tadd+jtf+7]+=tempf[ji+7];
        tfsI[tadd+jtf+8]+=tempf[ji+8];
        tfsI[tadd+jtf+9]+=tempf[ji+9];
        tfsI[tadd+jtf+10]+=tempf[ji+10];
        tfsI[tadd+jtf+11]+=tempf[ji+11];
        tfsI[tadd+jtf+12]+=tempf[ji+12];
        tfsI[tadd+jtf+13]+=tempf[ji+13];
        tfsI[tadd+jtf+14]+=tempf[ji+14];
        tfsI[tadd+jtf+15]+=tempf[ji+15];
        tfsI[tadd+jtf+16]+=tempf[ji+16];
        tfsI[tadd+jtf+17]+=tempf[ji+17];
        tfsI[tadd+jtf+18]+=tempf[ji+18];
        tfsI[tadd+jtf+19]+=tempf[ji+19];
        tfsI[tadd+jtf+20]+=tempf[ji+20];
        tfsI[tadd+jtf+21]+=tempf[ji+21];
        tfsI[tadd+jtf+22]+=tempf[ji+22];
        tfsI[tadd+jtf+23]+=tempf[ji+23];
        tfsI[tadd+jtf+24]+=tempf[ji+24];
        tfsI[tadd+jtf+25]+=tempf[ji+25];
        tfsI[tadd+jtf+26]+=tempf[ji+26];
        tfsI[tadd+jtf+27]+=tempf[ji+27];
        tfsI[tadd+jtf+28]+=tempf[ji+28];
        tfsI[tadd+jtf+29]+=tempf[ji+29];
        tfsI[tadd+jtf+30]+=tempf[ji+30];
        tfsI[tadd+jtf+31]+=tempf[ji+31];
        tfsI[tadd+jtf+32]+=tempf[ji+32];
        tfsI[tadd+jtf+33]+=tempf[ji+33];
        tfsI[tadd+jtf+34]+=tempf[ji+34];
        tfsI[tadd+jtf+35]+=tempf[ji+35];
        tfsI[tadd+jtf+36]+=tempf[ji+36];
        tfsI[tadd+jtf+37]+=tempf[ji+37];
        tfsI[tadd+jtf+38]+=tempf[ji+38];
        tfsI[tadd+jtf+39]+=tempf[ji+39];
        tfsI[tadd+jtf+40]+=tempf[ji+40];
        tfsI[tadd+jtf+41]+=tempf[ji+41];
        tfsI[tadd+jtf+42]+=tempf[ji+42];
        tfsI[tadd+jtf+43]+=tempf[ji+43];
        tfsI[tadd+jtf+44]+=tempf[ji+44];
        tfsI[tadd+jtf+45]+=tempf[ji+45];
        tfsI[tadd+jtf+46]+=tempf[ji+46];
        tfsI[tadd+jtf+47]+=tempf[ji+47];
        tfsI[tadd+jtf+48]+=tempf[ji+48];
        tfsI[tadd+jtf+49]+=tempf[ji+49];

        ji+=50;
        jtf+=50;
        if(jtf>=recNum){
            jtf=0;
            ++count01[1&labes[jln2]];
            ++jln2;
            if(jln2>=readln)
                break;
            tadd=recNum&labes[jln2];
        }
    }
//    cout<<"进程t"<<proId<<"tfsI[1000]="<<tfsI[1000]<<endl;/////////////////

    if(proId==(proN-1)){
        int ttfsI[recNum*2+2];
        int l=recNum*2;
        for(int z=0;z<proN-1;++z){
            //等待一个进程结束
            pid_t pid=wait(NULL);
            int i=0;
            for(;i<proN-1;++i){
                if(proH[i]==pid)
                    break;
            }
//            cout<<"收集进程"<<i<<endl;
            //合并特征
            int startR=i*(recNum+1)*2*sizeof(int);
            memcpy(ttfsI,shm+startR,(recNum+1)*2*sizeof(int));
            for(int j=0;j<l;++j){
                tfsI[j]+=ttfsI[j];
            }
            count01[0]+=ttfsI[l];
            count01[1]+=ttfsI[l+1];
//            cout<<i<<":"<<ttfsI[l]<<","<<ttfsI[l]<<endl;
        }
        for(int i=0;i<recNum;++i){
            tfs[i]=tfsI[i]*1.0f/count01[0];//暂不考虑分母为0情况
        }
        int rl=recNum*2;
        for(int i=recNum;i<rl;++i){
            tfs[i]=tfsI[i]*1.0f/count01[1];//暂不考虑分母为0情况//位运算要加()，优先级太低
        }

        delete[] tempf;
        close(fd);
//        munmap((void *)datas,totalLn*recNum*49/8);
//        munmap((void *)shm,proN*(recNum+1)*2*sizeof(int)+1);

//        struct timeval tv;/////////////////////
//        gettimeofday(&tv, NULL);/////////////////
//        cout<<tv.tv_usec<<"进程t"<<proId<<"结束"<<endl;

        return recNum;
    }else{
        tfsI[2*recNum]=count01[0];
        tfsI[2*recNum+1]=count01[1];


        int startR=proId*(recNum+1)*2*sizeof(int);//每个进程都存(recNum+1)*2个数,前proN表示是否完成
//        cout<<"进程t"<<proId<<"储存起始"<<startR<<"终止"<<proN*recNum*2*sizeof(int)<<endl;
//        cout<<proId<<"储存："<<count01[0]<<","<<count01[1]<<endl;

        memcpy(shm+startR,tfsI,(recNum+1)*2*sizeof(int));
        munmap((void *)shm,proN*(recNum+1)*2*sizeof(int)+1);

//        shm[proId]=1;//!int不能直接附给char
        delete[] tempf;

        return -1;
    }
}
char* predict(char *testData,float *tfs,int size,int recNum,char *plabels){
//    int countR=(size)/(recNum*6);//行数，假设规格统一，每行刚好6000字符
//    char *plabels=new char [countR*2];
//    char *fcs=new char[5];//0.xxx
    int jd=0,jf=-1,jnow=0,jlb=0;
    float sum0=0,sum1=0;
    float tempf[4];
    float32x4_t f4t_sum0 = vdupq_n_f32(0.0);
    float32x4_t f4t_sum1 = vdupq_n_f32(0.0);
    float32x4_t f4t_tef,f4t_c;

    float32x4_t f4t_fts[recNum>>1];

    for(int i=0;i<(recNum>>1);i+=2){
        f4t_fts[i]=vld1q_f32(tfs+(i<<1));
        f4t_fts[i+1]=vld1q_f32(tfs+(i<<1)+recNum);
    }

    while(jd<size){
        tempf[0]=testData[jd+2]-'0';
        tempf[1]=testData[jd+8]-'0';
        tempf[2]=testData[jd+14]-'0';
        tempf[3]=testData[jd+20]-'0';
//        f4t_fts0=vld1q_f32(tfs+jnow);
//        f4t_fts1=vld1q_f32(tfs+recNum+jnow);
        f4t_tef=vld1q_f32(tempf);
        f4t_c=vsubq_f32(f4t_fts[jnow],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum0=vmlaq_f32(f4t_sum0,f4t_c,f4t_c);//计算 r = a+b*c
        f4t_c=vsubq_f32(f4t_fts[jnow+1],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum1=vmlaq_f32(f4t_sum1,f4t_c,f4t_c);//计算 r = a+b*c

        tempf[0]=testData[jd+26]-'0';
        tempf[1]=testData[jd+32]-'0';
        tempf[2]=testData[jd+38]-'0';
        tempf[3]=testData[jd+44]-'0';
//        f4t_fts0=vld1q_f32(tfs+jnow+4);
//        f4t_fts1=vld1q_f32(tfs+recNum+jnow+4);
        f4t_tef=vld1q_f32(tempf);
        f4t_c=vsubq_f32(f4t_fts[jnow+2],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum0=vmlaq_f32(f4t_sum0,f4t_c,f4t_c);//计算 r = a+b*c
        f4t_c=vsubq_f32(f4t_fts[jnow+3],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum1=vmlaq_f32(f4t_sum1,f4t_c,f4t_c);//计算 r = a+b*c

        tempf[0]=testData[jd+50]-'0';
        tempf[1]=testData[jd+56]-'0';
        tempf[2]=testData[jd+62]-'0';
        tempf[3]=testData[jd+68]-'0';
//        f4t_fts0=vld1q_f32(tfs+jnow+8);
//        f4t_fts1=vld1q_f32(tfs+recNum+jnow+8);
        f4t_tef=vld1q_f32(tempf);
        f4t_c=vsubq_f32(f4t_fts[jnow+4],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum0=vmlaq_f32(f4t_sum0,f4t_c,f4t_c);//计算 r = a+b*c
        f4t_c=vsubq_f32(f4t_fts[jnow+5],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum1=vmlaq_f32(f4t_sum1,f4t_c,f4t_c);//计算 r = a+b*c

        tempf[0]=testData[jd+74]-'0';
        tempf[1]=testData[jd+80]-'0';
        tempf[2]=testData[jd+86]-'0';
        tempf[3]=testData[jd+92]-'0';
//        f4t_fts0=vld1q_f32(tfs+jnow+12);
//        f4t_fts1=vld1q_f32(tfs+recNum+jnow+12);
        f4t_tef=vld1q_f32(tempf);
        f4t_c=vsubq_f32(f4t_fts[jnow+6],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum0=vmlaq_f32(f4t_sum0,f4t_c,f4t_c);//计算 r = a+b*c
        f4t_c=vsubq_f32(f4t_fts[jnow+7],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum1=vmlaq_f32(f4t_sum1,f4t_c,f4t_c);//计算 r = a+b*c

        tempf[0]=testData[jd+98]-'0';
        tempf[1]=testData[jd+104]-'0';
        tempf[2]=testData[jd+110]-'0';
        tempf[3]=testData[jd+116]-'0';
//        f4t_fts0=vld1q_f32(tfs+jnow+16);
//        f4t_fts1=vld1q_f32(tfs+recNum+jnow+16);
        f4t_tef=vld1q_f32(tempf);
        f4t_c=vsubq_f32(f4t_fts[jnow+8],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum0=vmlaq_f32(f4t_sum0,f4t_c,f4t_c);//计算 r = a+b*c
        f4t_c=vsubq_f32(f4t_fts[jnow+9],f4t_tef);//返回两个寄存器对应元素之差 r = a-b
        f4t_sum1=vmlaq_f32(f4t_sum1,f4t_c,f4t_c);//计算 r = a+b*c

        jnow+=10;
        if(jnow==(recNum>>1)){
            sum0=vgetq_lane_f32(f4t_sum0, 0)+vgetq_lane_f32(f4t_sum0, 1)+vgetq_lane_f32(f4t_sum0, 2)+vgetq_lane_f32(f4t_sum0, 3);
            sum1=vgetq_lane_f32(f4t_sum1, 0)+vgetq_lane_f32(f4t_sum1, 1)+vgetq_lane_f32(f4t_sum1, 2)+vgetq_lane_f32(f4t_sum1, 3);
            if(sum0<sum1){
                plabels[jlb]='0';
            }else{
                plabels[jlb]='1';
            }
            plabels[jlb+1]='\n';
            jlb+=2;
            f4t_sum0 = vdupq_n_f32(0.0);//清零方法有待改进
            f4t_sum1 = vdupq_n_f32(0.0);
            jnow=0;
        }
        jd+=120;
    }
    return plabels;
}
void loadPredict(const char* file,const char* pfile,float *tfs,int recNum) {
    //多进程
    int proN=16;//进程数
    int proId=0;
    int length  = FileSize(file);

    int countR=(length)/(recNum*6);//行数，若规格统一，每行刚好6000字符
    if((countR*recNum*6)!=length){
        countR+=1;
    }

    /*打开输出文件，直接写入*/
    int outFd = open(pfile, O_RDWR | O_CREAT, 0644);//1. 创建内存段
    if (outFd < 0) {
        printf("open <%s> failed: %s\n", outFd, strerror(errno));
    }
    ftruncate(outFd, countR*2);//2.设置共享内存大小
    void *addr=mmap(NULL,countR*2, PROT_READ | PROT_WRITE, MAP_SHARED, outFd, 0); //mmap映射系统内存池到进程内存
    if (MAP_FAILED == addr) {
        printf("mmap: %s\n", strerror(errno));
    }
    char *shm=(char *)addr;
    /**/
    /*分配进程相应任务量,以行为单位,5x,4x...2x...2x*/
    int proAddR[proN];//起始行
    int bc=countR/(8+(proN-2)*5/2);
    proAddR[0]=0;
    proAddR[1]=bc*5;
    proAddR[2]=bc*9;
    for(int i=3;i<proN/2;++i){
        proAddR[i]=bc*3+proAddR[i-1];//每个都少算一个bc，因为需要留给父进程也就是最后一个进程更多数据
    }
    for(int i=proN/2;i<proN;++i){
        proAddR[i]=bc*2+proAddR[i-1];//每个都少算一个bc，因为需要留给父进程也就是最后一个进程更多数据
    }
    int realSize;//真实读入量
//    for(int i=0;i<proN;++i){
//        cout<<proAddR[i]<<" ";
//    }
//    cout<<countR<<endl;/////////////
    /**/

    int fd=open(file,O_RDONLY);//O_RDONLY只读
    if(fd==-1){
        printf("%s\n",strerror(errno));
        return;
    }
    void *addr2=mmap(NULL,length, PROT_READ, MAP_PRIVATE, fd, 0); //mmap映射系统内存池到进程内存
    if (MAP_FAILED == addr2) {
        printf("mmap: %s\n", strerror(errno));
    }
    char *readf=(char *)addr2;

//    int cpus = 0;
//    cpus = sysconf(_SC_NPROCESSORS_CONF);

//    char *testData=new char[readSize];
    while(true){
        //读数据
        if(proId==(proN-1)){
            realSize=length-proAddR[proId]*6*recNum;
//            if((proId+1)*readSize>length){
//                realSize=length-proId*readSize;
//            }else{
//                realSize=readSize;
//            }
//            memcpy(testData,readf+proId*readSize,realSize);//最后一个进程没有读入
//            munmap((void *)readf,length);
//            close(fd);//文件句柄应该只有一个，不能多人读//先可运行，一会测试都关闭
            break;
        }else{
            realSize=(proAddR[proId+1]-proAddR[proId])*6*recNum;
//            memcpy(testData,readf+proId*readSize,realSize);
        }
        //开进程
        pid_t fpid=fork();//进程号
        if (fpid < 0)
            printf("error in fork!");
        else if (fpid == 0) {//子进程
//            setCpu(proId%cpus);////////////////
            break;
        }else {//父进程
        }
        proId+=1;
    }
//    struct timeval tv0;/////////////////////
//    gettimeofday(&tv0, NULL);/////////////////
//    cout<<tv0.tv_usec<<"进程"<<proId<<"开始"<<endl;
    //预测
//    cout<<"进程"<<proId<<",真实读入"<<realSize<<",真实读入行"<<realSize/(6*recNum)<<endl;///////////
//    for(int i=0;i<100;++i){
//        cout<<testData[i]<<" ";
//    }
//    cout<<"进程"<<proId<<endl;
    char *testData=readf+proAddR[proId]*6*recNum;
    char *plabels=shm+proAddR[proId]*2;
    plabels=predict(testData,tfs,realSize,recNum,plabels);
    //传递结果
//    memcpy(shm+proId*readR*2,result,(realSize/(6*recNum)*2));
//    delete [] result;
    //整合结果
    if(proId==(proN-1)){
//        munmap((void *)shm,countR*2);//也可以不用释放，进程结束，自动释放
//        close(outFd);
    }
//    struct timeval tv;/////////////////////
//    gettimeofday(&tv, NULL);/////////////////
//    cout<<tv.tv_usec<<"进程"<<proId<<"结束"<<endl;
}
int main(int argc, char *argv[])
{
    float tfs[2020];//1,0特征和

//    string trainFile = "/data/train_data.txt";
//    string testFile = "/data/test_data.txt";
//    string resultFile = "/projects/student/result.txt";

//    ///////////////////////////////
//    string root_path="E:/Datas/HWC";
////    string root_path="/root/data";
//    trainFile=root_path+trainFile;
//    testFile=root_path+testFile;
//    resultFile=root_path+resultFile;
//    /////////////////////

    int recNum=loadTrain("/data/train_data.txt",tfs);
    if(recNum<0){
        return 0;
    }
    loadPredict("/data/test_data.txt","/projects/student/result.txt",tfs,recNum);
    return 0;
}
//g++ -O3 Main3.cpp -o test -lpthread