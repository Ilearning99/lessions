#include<iostream>
#include<cstring>
#include<queue>
#include<cstdio>
using namespace std;
#define maxn 200010

int steps[maxn];
queue<int>que;

int main()
{
    int N,K;
    while(scanf("%d%d",&N,&K)!=EOF)
    {
        while(!que.empty())que.pop();
        memset(steps,-1,sizeof(steps));
        steps[N]=0;
        que.push(N);
        int ans=-1;
        while(!que.empty())
        {
            int f=que.front();
            que.pop();
            if(f==K)
            {
                ans=steps[f];
                break;
            }
            if(f-1>=0&&steps[f-1]==-1)
            {
                steps[f-1]=steps[f]+1;
                que.push(f-1);
            }
            if(f+1<maxn&&steps[f+1]==-1)
            {
                steps[f+1]=steps[f]+1;
                que.push(f+1);
            }
            if(2*f<maxn&&steps[2*f]==-1)
            {
                steps[2*f]=steps[f]+1;
                que.push(2*f);
            }
        }
        printf("%d\n",ans);
    }
    return 0;
}