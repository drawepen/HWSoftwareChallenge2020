#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <unistd.h>///////////linux
#include<string.h>
#include<sys/stat.h>
#include<fcntl.h>
//#include<io.h>
#include "set"

using namespace std;
clock_t startTime;
/**
 *
 * -test文件应该没有符号，格式统一、、
 * -train文件有未归一化数据，存在-号和>1
 *-扩展文件
 */
//加载训练数据并训练
int loadTrain(const char* file, float tfs[]) {
    int recNum=1000;
    int readln=1500;//读数据行数
    int readSize=7*recNum*(readln+1);//多读一行，第一行用于判断维度
    char *datas=new char[readSize];
    float *tempf=new float[readln*recNum];
    int labes[readln];

    memset(labes,0,readln*sizeof(int));
    int jf=-1,jd=0,jln=0;
    char fcs[5];//0.xxx
    int fd=open(file,O_RDONLY);//O_RDONLY只读
    if(fd==-1){
        printf("%s\n",strerror(errno));
    }else{
        int re=read(fd,datas,readSize);
        close(fd);
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
    memset(tfs,0,2*recNum*sizeof(float));//大小指的是字节大小
    int jnow=0;
    while(true){
        if(datas[jd]=='-'){
            memcpy(fcs,datas+jd+1,5);
            tempf[++jf]=-atof(fcs);//字符数组转浮点数
            jd+=7;
        }else{
            memcpy(fcs, datas + jd, 5);
            tempf[++jf]= atof(fcs);//字符数组转浮点数
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
        tfs[tadd+jtf+2]+=tempf[ji+2];;
        tfs[tadd+jtf+3]+=tempf[ji+3];
        tfs[tadd+jtf+4]+=tempf[ji+4];
        tfs[tadd+jtf+5]+=tempf[ji+5];;
        tfs[tadd+jtf+6]+=tempf[ji+6];
        tfs[tadd+jtf+7]+=tempf[ji+7];
        tfs[tadd+jtf+8]+=tempf[ji+8];
        tfs[tadd+jtf+9]+=tempf[ji+9];
        ji+=10;
        jtf+=10;
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

    delete[] datas;//**释放费时，提交时可尝试不释放
    delete[] tempf;//**

    return recNum;
}

char* loadPredict(const char* file,long *size) {
    int fd=open(file,O_RDONLY);//O_RDONLY只读
    char *testData=nullptr;
    if(fd==-1){
        printf("%s\n",strerror(errno));
    }else{
        long length=0;
        long oneSize=12000000;//初始
        int nTl=oneSize;
        char *temp=new char[oneSize];
        testData=new char[nTl];
        long tsize=read(fd,temp,(int)oneSize);
        while(tsize>0){
            length+=tsize;
            char *tt;
            if(length>nTl){
                tt=testData;
                testData=new char[3*nTl];
                memcpy(testData,tt,length-tsize);
            }
            memcpy(testData+(length-tsize),temp,tsize);
            if(length>nTl){
                char *te=temp;
                temp=tt;
                oneSize=nTl;//逐步大量读
                nTl*=3;
                delete[] te;
            }
            tsize=read(fd,temp,(int)oneSize);
        }

        close(fd);
        *size=length;
        delete[] temp;
    }
    cout<<(*size);////////////
//    //////////////////////////////
//    for(int i=0;i<7;++i){
//        FILE *fp = fopen(file,"a+"); //这一行代表创建txt文件
//        fprintf(fp,testData,size);
//        fclose(fp);
//    }
//    ///////////////////////////

    return testData;
}
char* predict(char *testData,float *tfs,long size,int recNum){
    float *tempf=new float[recNum];
    int countR=(size)/(recNum*6);//行数，若规格统一，每行刚好6000字符
    if(testData[size-1]!='\n'){
        countR+=1;
    }
    char *plabels=new char [countR*2];
    char *fcs=new char[5];//0.xxx
    int jd=0,jf=-1,jnow=0,jlb=0;
    float sum0=0,sum1=0;
    float pow;
    while(jd<size){
        memcpy(fcs,testData+jd,5);
        tempf[++jf]= atof(fcs);//字符数组转浮点数
        pow=tfs[jnow]-tempf[jf];
        sum0+=pow*pow;

        if(++jnow==recNum){
            jf=0;
            for(int i=0;i<recNum;++i){
                pow=tfs[i+recNum]-tempf[i];
                sum1+=pow*pow;
            }
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
void save(const char* file,char * str,int l){
    FILE *fp = fopen(file,"w"); //这一行代表创建txt文件
    fprintf(fp,str,l);
    fclose(fp);
//    int fd=open(file,O_WRONLY,O_CREAT);//!!!无法创建文件？？？
//    int re=write(fd,str,l);
//    close(fd);
}
int main(int argc, char *argv[])
{
    startTime=clock();//TEST
    float tfs[2020];//1,0特征和

    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string resultFile = "/projects/student/result.txt";

    ///////////////////////////////
    string root_path="E:/Datas/HWC";
//    string root_path="/root/data";
    trainFile=root_path+trainFile;
    testFile=root_path+testFile;
    resultFile=root_path+resultFile;
    /////////////////////

    int recNum=loadTrain(trainFile.c_str(),tfs);


    long *size = new long(0);
    char *testData = loadPredict(testFile.c_str(),size);

    int countR=(*size)/(recNum*6);//行数，若规格统一，每行刚好6000字符
    if(testData[*size-1]!='\n'){
        countR+=1;
    }
    char *result=predict(testData,tfs,*size,recNum);
    save(resultFile.c_str(),result,countR*2);

    delete size;
    delete[] testData;
    delete [] result;
    return 0;
}
//g++ -O3 Main2.cpp -o test -lpthread