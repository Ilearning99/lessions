#include<iostream>
#include<cstring>
#include<cmath>
using namespace std;
#define maxn 1000
#define maxm 1000
#define eps 1e-8

double co[maxm][maxn];
bool isbase[maxn];
int bases[maxn];
int n,m;

bool check(int i, int pos)
{
    int j;
    for(j=1;j<=m&&j!=pos;j++)
    {
        if(abs(co[j][i])>eps)
        {
            return false;
        }
    }
    return true;
}

void gao1(int index, int pos)
{
    double tmp = co[pos][index];
    for(i=0;i<=n;i++)
    {
        co[pos][i]/=tmp;
    }
    for(j=0;j<=m&&j!=pos;j++)
    {
        tmp=co[j][index];
        for(i=0;i<=n;i++)
        {
            co[j][i]-=co[pos][i]*tmp;
        }
    }
}

void display()
{
    int i,j;
    puts("-------------------------------------------------------")
    for(i=0;i<=m;i++)
    {
        for(j=0;j<=n;j++)
        {
            print("%f ",co[i][j]);
        }
        puts("");
    }
    puts("-------------------------------------------------------")
}

void init()
{
    int i,j;
    for(j=1;j<=m;j++)
    {
        for(i=0;i<n&&isbase[i]==false;i++)
        {
            if(abs(co[j][i]-1.0)<eps)
            {
                if(check(i,j))
                {
                    isbase[i]=true;
                    bases[j]=i;
                    gao1(i,j);
                    break;
                }
            }
        }
    }
}

void gao()
{
    init();
    int i,index;
    bool flag=true;
    while(flag)
    {
        flag=false;
        for(i=0;i<n;i++)
        {
            if(co[0][i]>eps)
            {
                index=i;
            }
        }
        
    }
}

int main()
{
    int i,j;
    while(true)
    {
        puts("input n m");
        scanf("%d%d",&n,&m);
        memset(co,0,sizeof(co));
        memset(isbase,false,sizeof(isbase));
        //input optimization target
        for(j=0;j<=m;j++)
        {
            for(i=0;i<=n;i++)
            {
                scanf("%lf",&co[j][i]);
            }
        }
        gao();
    }
    return 0;
}