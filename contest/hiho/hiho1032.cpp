#include<cstdio>
#include<algorithm>
using namespace std;
#define maxn 1000010

char str[maxn];
char A[maxn<<1];
int f[maxn<<1];

int main()
{
	int T,t,ans,i,center,right;
	scanf("%d",&T);
	while(T--)
	{
		scanf("%s",str);
		A[0]='@';
		A[1]='#';
		t=2;
		ans=0;
		for(i=0;str[i];i++)
		{
			A[t++]=str[i];
			A[t++]='#';
		}
		center=0;
		right=0;
		A[t]='$';
		for(i=0;i<t;i++)
		{
			f[i]=0;
			if(i<right)
			{
				f[i]=min(right-i,f[2*center-i]);
			}
			while(A[i+f[i]+1]==A[i-f[i]-1])
				f[i]++;
			if(right<i+f[i]){
				right=i+f[i];
				center=i;
			}
			ans=max(ans,f[i]);
		}
		printf("%d\n",ans);
	}
	return 0;
}