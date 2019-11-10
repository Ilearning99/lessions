#include<iostream>
#include<cstdio>
using namespace std;

int m,n;
int graph[20][20];
int flip[20][20];
int tmpflip[20][20];
int tmp[20][20];

void gao1()
{
    int i,j;
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            flip[i][j]=tmpflip[i][j];
        }
    }
}

void gao2(int x,int y)
{
    int dir[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    tmp[x][y]^=1;
    int i;
    for(i=0;i<4;i++)
    {
        int px=x+dir[i][0];
        int py=y+dir[i][1];
        if(px>=0&&px<m&&py>=0&&py<n)
        {
            tmp[px][py]^=1;
        }
    }
}

int gao(int state)
{
    int i,j,ans=0;
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            tmp[i][j]=graph[i][j];
            tmpflip[i][j]=0;
        }
    }
    for(j=0;j<n;j++)
    {
        if((1<<j)&state)
        {
            tmpflip[0][j]=1;
            ans++;
            gao2(0,j);
        }
    }
    for(i=1;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            if(tmp[i-1][j]==1)
            {
                ans++;
                tmpflip[i][j]=1;
                gao2(i,j);
            }
        }
    }
    for(j=0;j<n;j++)
    {
        if(tmp[m-1][j]==1)
            return -1;
    }
    return ans;
}

int main()
{
    int i,j,ans,tmp;
    while(scanf("%d%d",&m,&n)!=EOF)
    {
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
            {
                scanf("%d",&graph[i][j]);
            }
        }
        ans=-1;
        for(int state=0;state<(1<<n);state++)
        {
            tmp=gao(state);
            //printf("%d\n",tmp);
            if(tmp!=-1&&(ans==-1||ans>tmp))
            {
                ans=tmp;
                gao1();
            }
        }
        if(ans==-1)
        {
            puts("IMPOSSIBLE");
        }
        else
        {
            for(i=0;i<m;i++)
            {
                for(j=0;j<n;j++)
                {
                    if(j)
                        printf(" ");
                    printf("%d",flip[i][j]);
                }
                puts("");
            }
        }
    }
    return 0;
}