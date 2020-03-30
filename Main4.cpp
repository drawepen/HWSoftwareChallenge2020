#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <unistd.h>///////////linux
#include<string.h>
#include<sys/stat.h>
#include<fcntl.h>
#include "set"

using namespace std;
clock_t startTime;
/**
 *
 * -test文件应该没有符号，格式统一、、
 * -train文件有未归一化数据，存在-号和>1
 * -只使用一位小数
 * -特征分级为1,2,4,8，以便只用移位运算
 */

int FileSize(const char* fname)
{
    struct stat statbuf;
    if(stat(fname,&statbuf)==0)
        return statbuf.st_size;
    return -1;
}
//加载训练数据并训练
int loadTrain(const char* file, int tfs[]) {
    int recNum=1000;
    int readln=7000;//读数据行数
    int readSize=7*recNum*(readln+1);//多读一行，第一行用于判断维度
    char *datas=new char[readSize];
    int *tempf=new int[readln*recNum];
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
            tempf[++jf]='0'-fcs[2];//-atof(fcs);//字符数组转浮点数
            jd+=7;
        }else{
            memcpy(fcs, datas + jd, 5);
            tempf[++jf]=fcs[2]-'0';//atof(fcs);//字符数组转浮点数
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
    int max=tfs[0],min=tfs[0],sum=tfs[0];
    for(int i=1;i<recNum;++i){
        if(tfs[i]>max){
            max=tfs[i];
        }else if(tfs[i]<min){
            min=tfs[i];
        }
        sum+=tfs[i];
    }
    int jd2=sum/count01[0];
    int jd1=(min+jd2)/2;
    int jd3=(max+jd2)/2;
    for(int i=0;i<recNum;++i){
        if(tfs[i]<jd1){
            tfs[i]=0;
        }else if(tfs[i]<jd2){
            tfs[i]=1;
        }else if(tfs[i]<jd2){
            tfs[i]=2;
        }else{
            tfs[i]=3;
        }
    }
    max=tfs[recNum];min=tfs[recNum];sum=tfs[recNum];
    int l=recNum*2;
    for(int i=recNum+1;i<l;++i){
        if(tfs[i]>max){
            max=tfs[i];
        }else if(tfs[i]<min){
            min=tfs[i];
        }
        sum+=tfs[i];
    }
    jd2=sum/count01[1];
    jd1=(min+jd2)/2;
    jd3=(max+jd2)/2;
    for(int i=recNum;i<l;++i){
        if(tfs[i]<jd1){
            tfs[i]=0;
        }else if(tfs[i]<jd2){
            tfs[i]=1;
        }else if(tfs[i]<jd2){
            tfs[i]=2;
        }else{
            tfs[i]=3;
        }
    }

    delete[] datas;//**释放费时，提交时可尝试不释放
    delete[] tempf;//**

    return recNum;
}

char* loadPredict(const char* file,long *size) {
    long length = FileSize(file);//要打开文件前读取，否则读不来
    int fd=open(file,O_RDONLY);//O_RDONLY只读
    char *testData=nullptr;
    if(fd==-1){
        printf("%s\n",strerror(errno));
    }else{
        testData=new char[(int)length];
        long tsize=read(fd,testData,(int)length);

        close(fd);
        *size=length;
    }
    return testData;
}
char* predict(char *testData,int *tfs,long size,int recNum){
    int countR=(size)/(recNum*6);//行数，若规格统一，每行刚好6000字符
    if(testData[size-1]!='\n'){
        countR+=1;
    }
    char *plabels=new char [countR*2];
    char *fcs=new char[5];//0.xxx
    int jd=0,jf=-1,jnow=0,jlb=0;
    float sum0=0,sum1=0;
    int tempf,pow;
    while(jd<size){
        memcpy(fcs,testData+jd,5);
        tempf= fcs[2]-'0';//atof(fcs);//字符数组转浮点数
        sum0+=tempf<<tfs[jnow];
        sum1+=tempf<<tfs[jnow+recNum];
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
    int tfs[2020];//1,0特征和

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
///////////////////
    float sum=0;
    for(int i=0;i<1000;++i){
        sum+=tfs[i];
    }
    cout<<sum<<endl;
    sum=0;
    for(int i=1000;i<2000;++i){
        sum+=tfs[i];
    }
    cout<<sum<<endl;
//////////////////////


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