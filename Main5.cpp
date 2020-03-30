#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <unistd.h>///////////linux
#include<string.h>
#include<sys/stat.h>
#include<fcntl.h>
#include "set"
#include <sys/mman.h>

using namespace std;
/**
 *
 * -test文件应该没有符号，格式统一、、
 * -train文件有未归一化数据，存在-号和>1
 * -去除含-号的数据/////不理想
 * -多进程
 * -只使用一位小数
 * -特征分级为1,2,4,8，以便只用移位运算//////失败
 */

int FileSize(const char* fname)
{
    struct stat statbuf;//获取的文件大小含索引，不能用
    if(stat(fname,&statbuf)==0)
        return statbuf.st_size;
    return -1;
}
//加载训练数据并训练
int loadTrain(const char* file, float tfs[]) {
    int recNum=1000;
    int readln=7000;//读数据行数
    int readSize=7*recNum*(readln+1);//
    char *datas;
    int *tempf=new int[readln*recNum];
    int labes[readln];

    memset(labes,0,readln*sizeof(int));
    int jf=-1,jd=0,jln=0;
//    char fcs[5];//0.xxx
    int fd=open(file,O_RDONLY);//O_RDONLY只读
    if(fd==-1){
        printf("%s\n",strerror(errno));
    }else{
        void *addr2=mmap(NULL,readSize, PROT_READ, MAP_PRIVATE, fd, 0); //mmap映射系统内存池到进程内存
        if (MAP_FAILED == addr2) {
            printf("mmap: %s\n", strerror(errno));
        }
        datas=(char *)addr2;

//        int re=read(fd,datas,readSize);
//        close(fd);
    }
    //确定维度
    recNum=0;
    while(datas[jd]!='\n'){
        if(datas[jd]==','){
            recNum+=1;
        }
        jd+=1;
    }
    ++jd;//从下一行开始记
    /*
     * 格式必须统一才可读取
     */
//    cout<<"1";////////////
    memset(tfs,0,2*recNum*sizeof(float));//大小指的是字节大小
    int jnow=0;
    while(jd<readSize){
        if(datas[jd]=='-'){
            tempf[++jf]='0'-datas[jd+3];//字符数组转浮点数
            jd+=7;
        }else{
            tempf[++jf]= datas[jd+2]-'0';//字符数组转浮点数
            jd += 6;
        }
        if(++jnow==recNum){
            if(datas[jd]-'0'){
                labes[jln]=0x0fffffff;
            }
            jln+=1;
            if(jln>=readln){
                break;
            }
            jd+=2;
            jnow=0;
        }
    }
    int sl=(recNum)*readln;
    int ji=0,jtf=0,jln2=0;
    int tadd=recNum&labes[jln2];
    int count01[2]={0,0};

    while(ji<sl){
        tfs[tadd+jtf]  +=tempf[ji];
        tfs[tadd+jtf+1]+=tempf[ji+1];
        tfs[tadd+jtf+2]+=tempf[ji+2];
        tfs[tadd+jtf+3]+=tempf[ji+3];
        tfs[tadd+jtf+4]+=tempf[ji+4];
        tfs[tadd+jtf+5]+=tempf[ji+5];
        tfs[tadd+jtf+6]+=tempf[ji+6];
        tfs[tadd+jtf+7]+=tempf[ji+7];
        tfs[tadd+jtf+8]+=tempf[ji+8];
        tfs[tadd+jtf+9]+=tempf[ji+9];
        tfs[tadd+jtf+10]+=tempf[ji+10];
        tfs[tadd+jtf+11]+=tempf[ji+11];
        tfs[tadd+jtf+12]+=tempf[ji+12];
        tfs[tadd+jtf+13]+=tempf[ji+13];
        tfs[tadd+jtf+14]+=tempf[ji+14];
        tfs[tadd+jtf+15]+=tempf[ji+15];
        tfs[tadd+jtf+16]+=tempf[ji+16];
        tfs[tadd+jtf+17]+=tempf[ji+17];
        tfs[tadd+jtf+18]+=tempf[ji+18];
        tfs[tadd+jtf+19]+=tempf[ji+19];
        ji+=20;
        jtf+=20;
        if(jtf>=recNum){
            jtf=0;
            ++count01[1&labes[jln2]];
            ++jln2;
            if(jln2>=readln)
                break;
            tadd=recNum&labes[jln2];
        }
    }
    for(int i=0;i<recNum;++i){
        tfs[i]/=count01[0];//暂不考虑分母为0情况
    }
    int l=recNum*2;
    for(int i=recNum;i<l;++i){
        tfs[i]/=count01[1];
    }

//    delete[] datas;
    delete[] tempf;

    ///
    close(fd);
    munmap((void *)datas,readSize);
    ///
    return recNum;
}
char* predict(char *testData,float *tfs,int size,int recNum,char *plabels){
    int countR=(size)/(recNum*6);//行数，假设规格统一，每行刚好6000字符
//    char *plabels=new char [countR*2];
//    char *fcs=new char[5];//0.xxx
    int jd=0,jf=-1,jnow=0,jlb=0;
    float sum0=0,sum1=0;
    float tempf,pow;
    while(jd<size){
//        memcpy(fcs,testData+jd,5);
        tempf= testData[jd+2]-'0';//字符数组转浮点数
        pow=tfs[jnow]-tempf;
        sum0+=pow*pow;
        pow=tfs[jnow+recNum]-tempf;
        sum1+=pow*pow;
        if(++jnow==recNum){
            if(sum0<sum1){
                plabels[jlb]='0';
            }else{
                plabels[jlb]='1';
            }
            plabels[jlb+1]='\n';
            jlb+=2;
            sum0=0;
            sum1=0;
            jnow=0;
        }
        jd+=6;
    }
    return plabels;
}
void loadPredict(const char* file,const char* pfile,float *tfs,int recNum) {
    //多进程
    int proN=10;//进程数
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

    int readR=countR/proN;//预计读入行数
    if(readR*proN!=countR){
        readR+=1;
    }
    int readSize=readR*6*recNum;
    int realSize;//真实读入量

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

//    char *testData=new char[readSize];
    while(true){
        //读数据
        if(proId==(proN-1)){
            if((proId+1)*readSize>length){
                realSize=length-proId*readSize;
            }else{
                realSize=readSize;
            }
//            memcpy(testData,readf+proId*readSize,realSize);//最后一个进程没有读入
//            munmap((void *)readf,length);
//            close(fd);//文件句柄应该只有一个，不能多人读//先可运行，一会测试都关闭
            break;
        }else{
            realSize=readSize;
//            memcpy(testData,readf+proId*readSize,realSize);
        }
        //开进程
        pid_t fpid=fork();//进程号
        if (fpid < 0)
            printf("error in fork!");
        else if (fpid == 0) {//子进程
            break;
        }else {//父进程
        }
        proId+=1;
    }
    //预测
//    cout<<"进程"<<proId<<",真实读入"<<realSize<<",真实读入行"<<realSize/(6*recNum)<<endl;///////////
//    for(int i=0;i<100;++i){
//        cout<<testData[i]<<" ";
//    }
//    cout<<"进程"<<proId<<endl;
    char *testData=readf+proId*readSize;
    char *plabels=shm+proId*readR*2;
    plabels=predict(testData,tfs,realSize,recNum,plabels);
    //传递结果
//    memcpy(shm+proId*readR*2,result,(realSize/(6*recNum)*2));
//    delete [] result;
    //整合结果
    if(proId==(proN-1)){
//        munmap((void *)shm,countR*2);//也可以不用释放，进程结束，自动释放
//        close(outFd);
    }
}
int main(int argc, char *argv[])
{
    float tfs[2020];//1,0特征和

    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string resultFile = "/projects/student/result.txt";

//    ///////////////////////////////
//    string root_path="E:/Datas/HWC";
////    string root_path="/root/data";
//    trainFile=root_path+trainFile;
//    testFile=root_path+testFile;
//    resultFile=root_path+resultFile;
//    /////////////////////

    int recNum=loadTrain(trainFile.c_str(),tfs);
    loadPredict(testFile.c_str(),resultFile.c_str(),tfs,recNum);
    return 0;
}
//g++ -O3 Main3.cpp -o test -lpthread