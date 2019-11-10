#include<iostream>
#include<cstring>
#include<queue>
#include<cstdio>
using namespace std;

char graph[33][33][33];

struct node
{
    int l,r,c;
};

node s,e;

int steps[33][33][33];

queue<node> que;
int L,R,C;

bool check(node t)
{
    if(t.l<0||t.l>=L)
        return false;
    if(t.r<0||t.r>=R)
        return false;
    if(t.c<0||t.c>=C)
        return false;
    return true;
}

void input(node t,int pre)
{
    //printf("%d\t%d\t%d\n",t.l,t.r,t.c);
    if(check(t)&&steps[t.l][t.r][t.c]==-1&&graph[t.l][t.r][t.c]!='#')
    {
        //printf("%d\t%d\t%d\t%c\n",t.l,t.r,t.c,graph[t.l][t.r][t.c]);
        steps[t.l][t.r][t.c]=pre+1;
        que.push(t);
    }
}

int gao()
{
    while(!que.empty())
    {
        node f=que.front();
        que.pop();
        //printf("%d\n",steps[f.l][f.r][f.c]);
        if(f.l==e.l&&f.r==e.r&&f.c==e.c)
            return steps[f.l][f.r][f.c];
        node t;
        t.l=f.l;
        t.r=f.r;
        t.c=f.c;
        int pre=steps[f.l][f.r][f.c];
        t.l=f.l-1;
        input(t,pre);
        t.l=f.l+1;
        input(t,pre);
        t.l=f.l;
        t.r=f.r-1;
        input(t,pre);
        t.r=f.r+1;
        input(t,pre);
        t.r=f.r;
        t.c=f.c-1;
        input(t,pre);
        t.c=f.c+1;
        input(t,pre);
    }
    return -1;
}

int main()
{
    while(true)
    {
        scanf("%d%d%d",&L,&R,&C);
        if(L==0||R==0||C==0)
            break;
        memset(steps,-1,sizeof(steps));
        for(int i=0;i<L;i++)
            for(int j=0;j<R;j++)
                scanf("%s",graph[i][j]);
        while(!que.empty())
            que.pop();
        for(int i=0;i<L;i++)
            for(int j=0;j<R;j++)
                for(int k=0;k<C;k++)
                    if(graph[i][j][k]=='S')
                    {
                        s.l=i;
                        s.r=j;
                        s.c=k;
                        steps[i][j][k]=0;
                        que.push(s);
                    }
                    else if(graph[i][j][k]=='E')
                    {
                        e.l=i;
                        e.r=j;
                        e.c=k;
                    }
        int ans=gao();
        if(ans==-1)
            puts("Trapped!");
        else
            printf("Escaped in %d minute(s).\n",ans);
    }
    return 0;
}