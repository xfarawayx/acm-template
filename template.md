## 快速输入输出

### 仅输入

```cpp
namespace io {
	const int SIZE=(1<<21)+1;
	char ibuf[SIZE],*iS,*iT,c;
	#define gc()(iS==iT?(iT=(iS=ibuf)+fread(ibuf,1,SIZE,stdin),(iS==iT?EOF:*iS++)):*iS++)
	inline int gi (){
		int x=0,f=1;
		for(c=gc();c<'0'||c>'9';c=gc())if(c=='-')f=-1;
		for(;c<='9'&&c>='0';c=gc()) x=(x<<1)+(x<<3)+(c&15); return x*f;
	}
} using io::gi;
```

### 输入+输出

```cpp
namespace io
{
	const int SIZE = (1 << 21) + 1;
	char ibuf[SIZE],*iS,*iT,obuf[SIZE],*oS=obuf,*oT=oS+SIZE-1,c,qu[55]; int qr;
	#define gc()(iS==iT?(iT=(iS=ibuf)+fread(ibuf,1,SIZE,stdin),(iS==iT?EOF:*iS++)):*iS++)
	inline int gi()
	{
		int x=0,f=1;
		for(c=gc();c<'0'||c>'9';c=gc())if(c=='-')f=-1;
		for(;c<='9'&&c>='0';c=gc()) x=(x<<1)+(x<<3)+(c&15); return x*f;
	}
	inline void flush()
	{
		fwrite (obuf, 1, oS - obuf, stdout);
		oS = obuf;
	}
	inline void putc (char x)
	{
		*oS ++ = x;
		if (oS == oT) flush ();
	}
	template <class I>
	inline void print (I x)
	{
		if (!x) putc ('0'); if (x < 0) putc ('-'), x = -x;
		while (x) qu[++ qr] = x % 10 + '0',  x /= 10;
		while (qr) putc (qu[qr --]);
	}
	struct Flusher_ {~Flusher_(){flush();}}io_flusher_;
}
```

## 字符串

### Trie 树 + AC自动机

用途：多模字符串匹配；建 fail 树，子树外为子树内串的子串。

```cpp
void insert(char* s)
{
	int u=1;
	for(int i=1;i<=n;++i)
	{
		if(!ch[u][s[i]]) ch[u][s[i]]=++cnt;
		u=ch[u][s[i]];
	}
	ed[u]=true;
}
void ac_build()
{
	queue<int> q; q.push(1);
	while(!q.empty())
	{
		int u=q.front(); q.pop();
		for(int i=0;i<10;++i)
		{
			if(!ch[u][i]) ch[u][i]=(fail[u]?ch[fail[u]][i]:1);
			else fail[ch[u][i]]=(fail[u]?ch[fail[u]][i]:1),q.push(ch[u][i]);
		}
	}
}
```

### 后缀自动机

```cpp
int pre[N*3],len[N*3],ch[N*3][26],cnt=1,lst=1;
void extend(int c)
{
	int p=lst,np=++cnt;lst=np;
	len[np]=len[p]+1; //size[np]=1;
	for(;p&&!ch[p][c];p=pre[p]) ch[p][c]=np;
	if(!p) return void(pre[np]=1);
	int q=ch[p][c];
	if(len[q]==len[p]+1) return void(pre[np]=q);
	int nq=++cnt; len[nq]=len[p]+1;
	memcpy(ch[nq],ch[q],sizeof(ch[q]));
	pre[nq]=pre[q],pre[q]=pre[np]=nq;
	for(;ch[p][c]==q;p=pre[p]) ch[p][c]=nq;
}
```

广义后缀自动机（建一棵trie树）：

```cpp
void extend(int p, int c)
{
	if(ch[p][c]&&len[ch[p][c]]==len[p]+1) return void(np=ch[p][c]);
	np=++cnt,len[np]=len[p]+1;
	for(;p&&!ch[p][c];p=pre[p]) ch[p][c]=np;
	if(!p) return void(pre[np]=1);
	int q=ch[p][c];
	if(len[q]==len[p]+1) return void(pre[np]=q);
	int nq=++cnt; len[nq]=len[p]+1;
	memcpy(ch[nq],ch[q],sizeof(ch[q]));
	pre[nq]=pre[q],pre[q]=pre[np]=nq;
	for(;ch[p][c]==q;p=pre[p]) ch[p][c]=nq;
}
void bfs()
{
	queue<int> q;
	q.push(1),lst[1]=1;
	while(!q.empty())
	{
		int u=q.front(); q.pop();
		extend(lst[u],s[u]-'a'),pos[u]=np;
		for(auto v:e[u]) lst[v]=np,q.push(v);
	}
}
```

## 数论

### exgcd

求 $ax +by=\gcd(a,b)$ 的一组解。

```cpp
int exgcd(int a, int b, int& x, int& y)
{
	if(!b)
	{
		x=1,y=0;
		return a;
	}
	int r=exgcd(b,a%b,x,y);
	int t=x; x=y; y=t-a/b*y;
	return r;
}
```

### 欧拉函数

求单个欧拉函数：

```cpp
int phi(int x)
{
	int ret=x;
	for(int i=2;i*i<=x;i++)
	{
		if(x%i==0) ret=ret/i*(i-1);
		while(x%i==0) x/=i;
	}
	if(x>1) ret=ret/x*(x-1);
	return ret;
}
```

线性筛素数、欧拉函数和莫比乌斯函数：

```cpp
bool p[M];
void init()
{
	phi[1]=mu[1]=1;
	for(int i=2;i<N;++i)
	{
		if(!p[i]) prime[++id]=i,mu[i]=-1,phi[i]=i-1;
		for(int j=1;j<N&&1ll*i*prime[j]<=N;++j)
		{
			p[i*prime[j]]=true;
			if(i%prime[j]==0)
			{
				phi[i*prime[j]]=phi[i]*prime[j];
				break;
			}
			mu[i*prime[j]]=-mu[i];
			phi[i*prime[j]]=phi[i]*(prime[j]-1);
		}
	}
}
```

### 大整数分解

包含素数判定 Millar-Rabin 算法和质因数分解 Pollard-Rho 算法。

```cpp
typedef unsigned long long ull;
typedef long long ll;
typedef __int128 LL;
typedef __uint128_t uLL;
namespace pollard
{
	const int pr[]={2,3,5,7,11,13,17,19,23,29,31,37},pt=12;
	ll po(ll x, ll y, ll p)
	{
		ll r=1;
		for(;y;y>>=1,x=(LL)x*x%p) if(y&1) r=(LL)r*x%p;
		return r;
	}
	ull gcd(ull x, ull y)
	{
		return y?gcd(y,x%y):x;
	}
	bool check(ll n)
	{
		if(n==1) return false;
		for(int i=0;i<pt;++i) if(n%pr[i]==0) return n==pr[i];
		ll r=n-1,s=0;
		while(~r&1) r>>=1,++s;
		for(int i=0;i<pt;++i)
		{
			ll x=po(pr[i],r,n);
			for(int j=1;j<=s&&x>1;++j)
			{
				ll y=(LL)x*x%n;
				if(y==1&&x!=n-1) return false;
				x=y;
			}
			if(x!=1) return false;
		}
		return true;
	}
	ull get(ull n)
	{
		ull y=rand(); int r=rand();
		for(int l=1;;l<<=1)
		{
			ull x=y;
			for(int i=0;i<l;++i) y=((uLL)y*y+r)%n;
			for(int k=0;k<l;k+=(1<<9))
			{
				int e=min(1<<9,l-k);
				ull g=1,z=y;
				for(int i=0;i<e;++i) y=((uLL)y*y+r)%n,g=(uLL)g*(y+n-x)%n;
				g=gcd(g,n);
				if(g==1) continue;
				if(g==n) for(g=1,y=z;g==1;) y=((uLL)y*y+r)%n,g=gcd(y+n-x,n);
				return g;
			}
		}
	}
	void rho(ll n, ll& a)
	{
		ll d=0;
		if(n<=a) return ;
		while(~n&1) n>>=1,a=2;
		if(n==1||check(n))
		{
			a=max(a,n);
			return ;
		}
		for(d=get(n);d==n;d=get(n));
		if(d<n/d) d=n/d;
		rho(d,a),rho(n/d,a);
	}
}
```

### BSGS 算法

$\texttt{bsgs(a,b)}$：返回 $a^x\equiv b \pmod {Mod})$  中的 $x$​。

```cpp
unordered_map<int,int> mp;
int bsgs(int a, int b)
{
	mp.clear();
	const int s=ceil(sqrt(Mod));
	for(int i=1,x=mul(a,b);i<=s;++i,x=mul(x,a)) mp[x]=i;
	for(int i=1,x=po(a,s),y=x;i<=s;++i,x=mul(x,y))
		if(mp[x]) return (mul(i,s)+Mod-mp[x])%Mod;
	return -1;
}
```

### EXCRT

求 $x\equiv a_i\pmod {b_i}$ 同余方程组的解。将所有 $a_i,b_i$ 带入，解保存在 $m$​ 中。可能需要验证解是否合法。

```cpp
void excrt(ll a, ll b) //x = a (mod b)
{
	ll c=((a-z)%b+b)%b,x,y;
	ll g=exgcd(m,b,x,y);
	if(c%g) {
		puts("-1");
		exit(0);
	}
	c/=g,b/=g;
	x=(x%b+b)%b;
	x=(__int128)x*c%b;
	z+=x*m;
	m*=b;
}
```

### 杜教筛

求欧拉函数的前缀和：

```cpp
int sub(int x, int y){
	return (x-y<0?x-y+Mod:x-y);
}
int sieve(ll n)
{
	if(n<=N) return phi[n];
	if(mp[n]) return mp[n];
	int ret=n*(n+1)/2%Mod; // n*(n+1)/2, modify needed 
	for(ll l=2,r;l<=n;l=r+1)
	{
		ll r=n/(n/l);
		ret=sub(res,(r-l+1)*solve(n/l)%Mod);
	}
	return mp[n]=ret;
}
```

求莫比乌斯函数的前缀和：

```cpp
int sieve(ll n)
{
	if(n<=N) return mu[n];
	if(mp[n]) return mp[n];
	int ret=1;
	for(ll l=2,r;l<=n;l=r+1)
	{
		ll r=n/(n/l);
		ret=sub(res,(r-l+1)*solve(n/l)%Mod);
	}
	return mp[n]=ret;
}
```

### Min25 筛

用途：求积性函数 $f(x)$​ 的前缀和。

第一阶段：构造（一个或多个）前缀和可快速计算的完全积性函数 $g(x)$ 使其在质数处的点值能通过运算能得到 $f(x)$ 在质数处的点值。令 $G(i,j)$ 表示前 $i$ 个数进行 $j$ 轮埃氏筛法的 $g(x)$ 和，则有 $G(i,j)=G(i,j-1)-[p_j\le i^2]\times g(p_j)\times ( G(\lfloor\frac{n}{p_j}\rfloor,j-1)-G(p_{j-1},j-1))$；

第二阶段：求出 $G(n,|P|)$​ 之后，令 $F(n,j)$​ 表示前 $n$​ 个数字，进行了 $j$​ 轮埃氏筛法，剩余的数字的 $f$​ 函数之和，则有 $F(n,j)=G(n,|p|)-\sum_{i=1}^{j-1}f(p_i)+\sum_{k> j}\sum_{e}(f(p_k^e)F(\lfloor\frac{n}{p_k^e}\rfloor,k+1)+f(p_k^{e+1}))$​​。

```cpp
//代码：求所有数因数个数的倒数和。构造 g(x)=1。
#include<bits/stdc++.h>
using namespace std;
const int N=1e5+5;
int n,p,m,ip,w[N],id[N],inv[233],pri[N],cnt,wcnt,id1[N],id2[N],g[N],sum[N];
bool b[N];
#define mul(x,y) (1ll*(x)*(y)%p)
inline int getid(int x)
{	return x<=m?id1[x]:id2[n/x];
}
inline int add(int x, int y)
{	return (x+y>=p?x+y-p:x+y);
}
inline int sub(int x, int y)
{	return (x-y<0?x-y+p:x-y);
}
int sieve(int n, int x)
{
	if(n<=1||pri[x]>n) return 0;
	int res=mul(ip,sub(g[getid(n)],sum[x-1])); // f(p) = (1/2)*g(p)
	for(int i=x;i<=cnt&&1ll*pri[i]*pri[i]<=n;++i)
		for(int e=1,now=pri[i];1ll*now*pri[i]<=n;++e,now*=pri[i])
			res=add(res,add(mul(inv[e+1],sieve(n/now,i+1)),inv[e+2]));
	return res;
}
int main()
{
	scanf("%d%d",&n,&p),ip=p+1>>1,m=sqrt(n);
	for(int i=2;i<=m;++i)
	{
		if(!b[i]) pri[++cnt]=i,sum[cnt]=add(sum[cnt-1],1);
		for(int j=1;j<=m&&i*pri[j]<=m;++j)
		{
			b[i*pri[j]]=true;
			if(i%pri[j]==0) break;
		}
	}
	inv[0]=inv[1]=1;
	for(int i=2;i<233;++i) inv[i]=mul(p-p/i,inv[p%i]);
	for(int i=1,j;i<=n;i=j+1)
	{
		j=n/(n/i),w[++wcnt]=n/i;
		g[wcnt]=n/i-1; //求 G(i,0)，即 g(i) 的前缀和。
		n/i<=m?id1[n/i]=wcnt:id2[j]=wcnt;
	}
	for(int j=1;j<=cnt;++j)
		for(int i=1;i<=wcnt&&1ll*pri[j]*pri[j]<=w[i];++i)
			g[i]=sub(g[i],mul(1,sub(g[getid(w[i]/pri[j])],sum[j-1]))); //求 G(i,j).
	printf("%d\n",add(sieve(n,1),1));
}
```

### Powerful Number

利用 Powerful Number 可以求部分积性函数 $F(x)$​ 的前缀和，优点是写的快。

构造一个积性函数 $G(x)$​，使得 $x$​ 为质数时 $G(x)=F(x)$​，且 $G(x)$​​​ 的前缀和可快速计算。

代码中：$\texttt{F(x,e)}$ 为 $F(x^e)$；$\texttt{G(x,e)}$ 为 $G(x^e)$（可递推计算）；$\texttt{sumG(x)}$ 为 $\sum_{i=1}^x G(i)$​。

```cpp
void dfs(int u, ll x, ll w)
{
	if(!w) return ;
	ans+=w*sumG(n/x);
	for(int i=u;i<=tot&&x<=n/pri[i]/pri[i];++i)
	{
		if(h[i].size()==1) h[i].push_back(0);
		ll y=pri[i];
		for(int e=2;x*y<=n/pri[i];++e)
		{
			y*=pri[i];
			if(h[i].size()==e)
			{
				ll f=F(pri[i],e),g=G(pri[i],1);
				for(int j=1;j<=e;++j) f-=g*h[i][e-j],g=G(pri[i],j+1);
				h[i].push_back(f);
			}
			dfs(i+1,x*y,w*h[i][e]);
		}
	}
}
ll solve()
{
	init(); //需筛质数。
	for(int i=0;i<=tot;++i) h[i].push_back(1);
	dfs(1,1,1); return ans;
}
```

## 数据结构

### 哈希表

```cpp
struct hash_map
{
	const int bi=(1<<14)-1;
	int tot,head[(1<<14)+5],nxt[N],to[N],wei[N];
	vector<int> rec;
	void adde(int a, int b)
	{
		nxt[++tot]=head[a&bi],head[a&bi]=tot,to[tot]=a,wei[tot]=b,rec.push_back(a&bi);
	}
	int query(int a)
	{
		for(int e=head[a&bi];e;e=nxt[e]) if(to[e]==a) return wei[e];
		return -1;
	}
	void clr()
	{
		tot=0;
		for(auto x:rec) head[x]=0;
		rec.clear();
	}
} ;
```

### ST 表

一维 RMQ：

```cpp
int f[N][25],lg[N];
void init(int n)
{
	for(int i=2;i<=n;i++) lg[i]=lg[i-1]+!(i&(i-1));
	for(int j=1;j<=lg[n];j++)
		for(int i=1;i+(1<<j)-1<=n;i++)
			f[i][j]=max(f[i][j-1],f[i+(1<<(j-1))][j-1]);
}
int query(int l, int r)
{
	int k=lg[r-l+1];
	return max(f[l][k],f[r-(1<<k)+1][k]);
}
```

二维 RMQ：

```cpp
int f[N][N][25],lg[N];
void init(int n, int m)
{
	int nm=max(n,m);
	for(int i=2;i<=nm;++i) lg[i]=lg[i-1]+!(i&(i-1));
	for(int k=1;k<=lg[nm];k++)
		for(int i=1;i+(1<<k)-1<=n;i++)
			for(int j=1;j+(1<<k)-1<=m;j++)
				f[i][j][k]=max(max(f[i][j][k-1],f[i+(1<<k-1)][j][k-1]),max(f[i+(1<<k-1)][j+(1<<k-1)][k-1],f[i][j+(1<<k-1)][k-1]));
}
int query(int x1, int y1, int x2, int y2)
{
	int l=lg[min(x2-x1+1,y2-y1+1)];
	int l2=(1<<l),ans=-1,lx2=x2,ly2=y2;
	x2-=l2-1; y2-=l2-1; 
	for(int x=x1;x+l2<=lx2;x+=l2) ans=max(ans,max(f[x][y1][l],f[x][y2][l]));
	for(int y=y1;y+l2<=ly2;y+=l2) ans=max(ans,max(f[x1][y][l],f[x2][y][l]));
	ans=max(ans,max(max(f[x1][y1][l],f[x1][y2][l]),max(f[x2][y1][l],f[x2][y2][l])));
	return ans;
}
```

### $O(n\log n)$ - O(1) LCA

```cpp
const int N=1e6+5,L=21; //N 开两倍
int f[L][N],dfn[N],rid[N],sze[N],fir[N],lg[N],dep[N],fa[N],tid,id;
vector<int> e[N];
struct ST
{
	#define cmp(x,y) (fir[(x)]<fir[(y)]?(x):(y))
	void build()
	{
		for(int i=2;i<=id;i++) lg[i]=lg[i>>1]+1;
		for(int j=1;(1<<j)<=id;j++)
			for (int i=1;i+(1<<(j-1))-1<=id;i++)
				f[j][i]=cmp(f[j-1][i],f[j-1][i+(1<<(j-1))]);
	}
	inline int lca(int u,int v)
	{
		int x=fir[u],y=fir[v];
		if(x>y) swap(x,y);
		int l=lg[y-x+1];
		return cmp(f[l][x],f[l][y-(1<<l)+1]);
	}
    #undef cmp
} rmq;
void dfs(int u)
{
	dfn[u]=++tid,rid[tid]=u,sze[u]=1;
	f[0][++id]=u,fir[u]=id;
	for(auto v:e[u]) if(v!=fa[u])
	{
		fa[v]=u,dep[v]=dep[u]+1,dfs(v);
		sze[u]+=sze[v],f[0][++id]=u;
	}
}
```

### Link-Cut Tree

```cpp
#define isnrt(x) (ch[fa[x]][0]==x||ch[fa[x]][1]==x)
void pushup(int x) { /*...*/ }
void pushdown(int x)
{
	if(rev[x])
	{
		rev[ch[x][0]]^=1,rev[ch[x][1]]^=1;
		swap(ch[x][0],ch[x][1]);
		rev[x]=0;
	}
}
void rotate(int x)
{
	int y=fa[x],z=fa[y];
	int k=ch[y][1]==x,w=ch[x][!k];
	if(isnrt(y)) ch[z][ch[z][1]==y]=x;
	ch[x][!k]=y,ch[y][k]=w;
	fa[w]=y,fa[y]=x,fa[x]=z;
	pushup(y),pushup(x);
}
void maintain(int x)
{
	if(isnrt(x)) maintain(fa[x]);
	pushdown(x);
}
void splay(int x)
{
	maintain(x);
	while(isnrt(x))
	{
		int y=fa[x],z=fa[y];
		if(isnrt(y)) (ch[z][1]==y)^(ch[y][1]==x)?rotate(x):rotate(y);
		rotate(x);
	}
	pushup(x); 
}
void access(int x) {int y=0; for(;x;y=x,x=fa[x]) splay(x),ch[x][1]=y,pushup(x); }
void makert(int x) { access(x),splay(x),rev[x]^=1; }
void link(int x, int y) { makert(x); fa[x]=y; }
void split(int x, int y) { makert(x),access(y),splay(y); }
void cut(int x, int y) { split(x,y); ch[y][0]=fa[x]=0; }
```

### KD 树

天使玩偶（插入一个点；查询某个点最近曼哈顿距离，带替罪羊树重构）

```cpp
const int N=2e6+5,Inf=1<<30;
inline void chmin(int& x, int y)
{
	x=(x<y?x:y);
}
inline void chmax(int& x, int y)
{
	x=(x>y?x:y);
}
struct node
{
	int d[2],mx[2],mn[2],l,r,size;
	void up(node x)
	{
		chmin(mn[0],x.mn[0]),chmax(mx[0],x.mx[0]);
		chmin(mn[1],x.mn[1]),chmax(mx[1],x.mx[1]);
	}
} st[N],q;
int rt,now,ans,n,m,qx,qy,v[N],tot,flag;
bool cmp(node x, node y)
{
	return x.d[now]<y.d[now];
}
void pushup(int x)
{
	if(st[x].l) st[x].up(st[st[x].l]);
	if(st[x].r) st[x].up(st[st[x].r]);
	st[x].size=1+st[st[x].l].size+st[st[x].r].size;
}
int build(int l=1, int r=n, bool x=0)
{
	int mid=l+r>>1; now=x;
	nth_element(st+l+1,st+mid+1,st+r+1,cmp);
	st[mid].mx[0]=st[mid].mn[0]=st[mid].d[0];
	st[mid].mx[1]=st[mid].mn[1]=st[mid].d[1];
	if(l!=mid) st[mid].l=build(l,mid-1,x^1);
	if(r!=mid) st[mid].r=build(mid+1,r,x^1);
	pushup(mid); return mid;
}
void dfs(int u)
{
	v[++tot]=u;
	if(st[u].l) dfs(st[u].l);
	if(st[u].r) dfs(st[u].r);
}
int rebuild(int l, int r, bool x)
{
	if(l>r) return 0;
	int mid=l+r>>1; now=x;
	nth_element(v+l+1,v+mid+1,v+r+1,[](int x, int y)
	{
		return cmp(st[x],st[y]);
	});
	int c=v[mid];
	if(l!=mid) st[c].l=rebuild(l,mid-1,x^1); else st[c].l=0;
	if(r!=mid) st[c].r=rebuild(mid+1,r,x^1); else st[c].r=0;
	pushup(c); return c;
}
void add(int& u, bool x=0, int fa=0)
{
	if(!u)
	{
		st[u=++n]=q;
		return ;
	}
	now=x; add(cmp(q,st[u])?st[u].l:st[u].r,x^1,u);
	now=x; pushup(u);
	if(!flag&&max(st[st[u].l].size,st[st[u].r].size)>st[u].size*0.75)
	{
		tot=0,dfs(u);
		if(fa)
		{
			if(st[fa].l==u) st[fa].l=rebuild(1,tot,x);
			else st[fa].r=rebuild(1,tot,x);
		}
		else rt=rebuild(1,tot,x);
	}
}
int getdist(int u)
{
	return max(0,st[u].mn[0]-qx)+max(0,qx-st[u].mx[0])+max(0,st[u].mn[1]-qy)+max(0,qy-st[u].mx[1]);
}
void query(int u=rt)
{
	ans=min(ans,abs(qx-st[u].d[0])+abs(qy-st[u].d[1]));
	int dl=(st[u].l?getdist(st[u].l):Inf),dr=(st[u].r?getdist(st[u].r):Inf);
	if(dl<dr)
	{
		if(dl<ans) query(st[u].l);
		if(dr<ans) query(st[u].r);
	}
	else
	{
		if(dr<ans) query(st[u].r);
		if(dl<ans) query(st[u].l);
	}
}
int main()
{
	n=gi(),m=gi();
	if(n==300000) flag=true; 
	for(int i=1;i<=n;++i) st[i].d[0]=gi(),st[i].d[1]=gi(),st[i].size=1;
	rt=build();
	while(m--)
	{
		int op=gi(),x=gi(),y=gi();
		if(op==1)
			q=(node){x,y,x,y,x,y,0,0,1},add(rt);
		if(op==2)
			ans=Inf,qx=x,qy=y,query(),print(ans);
	}
}
```

### 分块

```cpp
void update(int l, int r)
{
	for(;l<=r&&bel[l]==bel[l-1];l++) d[l]^=t;
	for(;l+B<r;l+=B) tag[bel[l]]^=t;
	for(;l<=r;l++) d[l]^=t;
}
void init()
{
	B=sqrt(n);
	for(int i=1;i<=n;++i) bel[i]=(i-1)/B+1;
	for(int i=1;i<=B;++i) br[i]=min(n,i*B);
}
```

### 莫队

```cpp
//带修莫队：小Z的袜子
int n,m,bel[N],a[N],tmp[N],tar[N],now[N],lst[N],ind1,ind2,l,r,t,ans[N],cnt[M],tot;
struct query
{
	int l,r,id,t;
} q[N];
bool operator < (query x, query y)
{
	if(bel[x.l]!=bel[y.l]) return bel[x.l]<bel[y.l];
	if(bel[x.r]!=bel[y.r]) return bel[x.r]<bel[y.r];
	return x.t<y.t;
}
void add(int x)
{
	if(!cnt[a[x]]) ++tot;
	++cnt[a[x]];
}
void del(int x)
{
	--cnt[a[x]];
	if(!cnt[a[x]]) --tot;
}
void upt(int t)
{
	if(l<=tar[t]&&tar[t]<=r) del(tar[t]);
	a[tar[t]]=now[t];
	if(l<=tar[t]&&tar[t]<=r) add(tar[t]);
}
void det(int t)
{
	if(l<=tar[t]&&tar[t]<=r) del(tar[t]);
	a[tar[t]]=lst[t];
	if(l<=tar[t]&&tar[t]<=r) add(tar[t]);
}
int main()
{
	n=gi(),m=gi();
	const int b=pow(n,2.0/3.0);
	for(int i=1;i<=n;i++) tmp[i]=a[i]=gi(),bel[i]=(i-1)/b+1;
	for(int i=1;i<=m;i++)
	{
		char op[2]; scanf("%s",op);
		int x=gi(),y=gi();
		if(op[0]=='Q') ++ind1,q[ind1]=(query){x,y,ind1,ind2};
		else
		{
			tar[++ind2]=x;
			now[ind2]=y; lst[ind2]=tmp[x];
			tmp[x]=y;
		}
	}
	sort(q+1,q+1+ind1);
	for(int i=1;i<=ind1;i++)
	{
		while(t<q[i].t) upt(++t);
		while(t>q[i].t) det(t--);
		while(r<q[i].r) add(++r);
		while(l>q[i].l) add(--l);
		while(r>q[i].r) del(r--);
		while(l<q[i].l) del(l++);
		ans[q[i].id]=tot;
	}
	for(int i=1;i<=ind1;i++) printf("%d\n",ans[i]);
}
```

### 珂朵莉树

```cpp
struct node {
	int l,r;
	mutable ll v;
	node(int L, int R=0, ll V=0) : l(L),r(R),v(V) {}
};
bool operator < (node x, node y) {
	return x.l<y.l;
} set<node> s;
typedef set<node>::iterator sit;
sit split(int p)
{
	sit it=s.lower_bound(node(p,0,0));
	if(it==s.begin()) return it;
	--it;
	int l=it->l,r=it->r; ll v=it->v;
	s.erase(it);
	s.insert(node(l,p-1,v));
	return s.insert(node(p,r,v)).first;
}
void assign(int l, int r, int w)
{
	sit itr=split(r+1),itl=split(l);
	s.erase(itl,itr);
	s.insert(node(l,r,w));
}
```

## 图论

### 最短路

堆优化 Dijkstra 算法。

```cpp
int dis[N];
typedef pair<int,int> pr;
priority_queue<pr,vector<pr>,greater<pr>> q;
int dijkstra(int s, int t)
{
	memset(dis,0x3f,sizeof(dis));
	dis[s]=0,q.push(make_pair(0,s));
	while(!q.empty())
	{
		int u=q.top().second,w=q.top().first;
		q.pop();
		if(dis[u]!=w) continue;
		for(auto v:e[u])
			if(dis[v]>dis[u]+wei[e])
			{
				dis[v]=dis[u]+wei[e];
				q.push(make_pair(dis[v],v));
			}
	}
	return dis[t];
}
```

### Tarjan 求强连通分量

```cpp
int n,dfn[N],low[N],tim,tp,st[N],id[N],scc;
bool ins[N],vis[N];
void tarjan(int u)
{
	dfn[u]=low[u]=++tim;
	ins[u]=true,st[++tp]=u;
	for(auto v:e[u])
	{
		if(!dfn[v])
		{
			tarjan(v);
			low[u]=min(low[u],low[v]);
		}
		else if(ins[v])
			low[u]=min(low[u],dfn[v]);
	}
	if(dfn[u]==low[u])
	{
		++scc; int v;
		do
		{
			v=st[tp--];
			ins[v]=false;
			id[v]=scc;
		} while(tp&&u!=v);
	}
}
```

### 求割点与桥

```cpp
void dfs(int u)
{
	low[u]=dfn[u]=++tid,sze[u]=1;
	int ch=0;
	for(auto v:e[u])
		if(!dfn[v])
		{
			++ch, dfs(v);
			sze[u]+=sze[v];
			if(dfn[u]<=low[v]&&u!=1) cut[u]=true;
			//if(dfn[u]<low[v]) --> (u,v) is a bridge. 
			low[u]=min(low[u],low[v]);
		}
		else low[u]=min(low[u],dfn[v]);
	if(u==1&&ch>1) cut[u]=true;
}
```

### 边双连通分量

边双：无割边

边双可从任一点进入，不经过重复边从任一点出。

```cpp
void tarjan(int u, int fa)
{
	low[u]=dfn[u]=++tid;
	st[++tp]=u;
	for(auto v:e1[u]) if(v!=fa)
	{
		if(!dfn[v]) tarjan(v,u),low[u]=min(low[u],low[v]);
		else if(!bel[v]) low[u]=min(low[u],dfn[v]);
	}
	if(low[u]==dfn[u])
	{
		++bid; int v;
		do
		{
			v=st[tp--],bel[v]=bid;
			++size[bid],sum[bid]+=w[v];
		} while(v!=u);
	}
}
```

### 点双连通分量+圆方树

点双：无割点

圆方树：对于每个点双联通分量（大小>=2）建立一个方点，向所有的点双内的点连边，用于处理有向图两点间路径的相关问题。

对于圆方树上的修改：我们强行让方点的权值不包括它的父亲(也就是只算它的儿子)，如果求解的时候 LCA 是方点，则额外计算一下方点父亲的权值，这样子每个圆点在修改的之后只需要向上修改给父亲方点啦！

```cpp
void tarjan(int u)
{
	dfn[u]=low[u]=++tim,stk[++tp]=u;
	for(auto v:v1[u])
	{
		if(!dfn[v])
		{
			tarjan(v);
			low[u]=min(low[u],low[v]);
			if(low[v]>=dfn[u])
			{
				v2[++id].push_back(u);
				v2[u].push_back(id);
				int x=0;
				do
				{
					x=stk[tp--];
					v2[id].push_back(x);
					v2[x].push_back(id);
				} while(x!=v);
			}
		}
		else low[u]=min(low[u],dfn[v]);
	}
}
```

### 二分图匹配（匈牙利算法）

```cpp
bool match(int u)
{
	for(auto v:e[u]) if(!vis[v])
	{
		vis[v]=true;
		if(!lnk[v]&&match(lnk[v]))
		{
			lnk[v]=u;
			return true;
		}
	}
	return false;
}
```

### 最大流

```cpp
int head[N],nxt[M],to[M],wei[M],lev[N],q[N],tot=1,n,m,s,t;
void addedge(int u, int v, int w)
{
	nxt[++tot]=head[u],head[u]=tot,to[tot]=v,wei[tot]=w;
	nxt[++tot]=head[v],head[v]=tot,to[tot]=u,wei[tot]=0;
}
bool bfs()
{
	memset(lev,-1,sizeof(lev));
	memcpy(cur,head,sizeof(head));
	int l=0,r=0;
	q[0]=s,lev[s]=1;
	while(l<=r)
	{
		int u=q[l++];
		for(int e=head[u];e;e=nxt[e])
			if(lev[to[e]]==-1&&wei[e])
			{
				lev[to[e]]=lev[u]+1;
				if(to[e]==t) return true;
				q[++r]=to[e];
			}
	}
	return false;
}
int dfs(int u, int mx)
{
	if(u==t) return mx;
	int l=mx;
	for(int& e=cur[u];e&&l;e=nxt[e])
		if(lev[to[e]]==lev[u]+1&&wei[e]>0)
		{
			int f=dfs(to[e],min(l,wei[e]));
			if(!f) --lev[to[e]];
			l-=f,wei[e]-=f,wei[e^1]+=f;
		}
	return mx-l;
}
int exec()
{
	int ans=0;
	while(bfs()) ans+=dfs(s,inf);
	return ans;
}
```

### 最小费用流

```cpp
int head[N],nxt[M],to[M],wei[M],cost[M],tot=1,cur[N],dis[N],n,m,s,t,ans,ans2;
bool vis[N];
void addedge(int u, int v, int w, int c)
{
	nxt[++tot]=head[u],head[u]=tot,to[tot]=v,wei[tot]=w,cost[tot]=c;
	nxt[++tot]=head[v],head[v]=tot,to[tot]=u,wei[tot]=0,cost[tot]=-c;
}
bool spfa()
{
//	memset(dis,0x3f,sizeof(dis));
	for(int i=0;i<=t;++i) dis[i]=inf;
	memset(vis,false,sizeof(vis));
	memcpy(cur,head,sizeof(head));
	int l=0,r=0;
	dis[t]=0,q[0]=t;
	while(l<=r)
	{
		int u=q[l++];
		vis[u]=false,q.pop();
		for(int e=head[u];e;e=nxt[e])
			if(wei[e^1]&&dis[to[e]]>dis[u]-cost[e])
			{
				dis[to[e]]=dis[u]-cost[e];
				if(!vis[to[e]]) vis[to[e]]=true,q[++r]=to[e];
			}
	}
	return dis[s]!=dis[0];
}
int dfs(int u, int mx)
{
	vis[u]=true;
	if(u==t) return mx;
	int l=mx;
	for(int& e=cur[u];e;e=nxt[e])
		if(!vis[to[e]]&&wei[e]>0&&dis[to[e]]==dis[u]-cost[e])
		{
			int f=dfs(to[e],min(l,wei[e]));
			ans+=f*cost[e];
			l-=f,wei[e]-=f,wei[e^1]+=f;
			if(!l) return mx;
		}
	return mx-l;
}
void exec()
{
	while(spfa())
		do
		{
			memset(vis,false,sizeof(vis));
			ans2+=dfs(s,inf);
		} while(vis[t]);
}
```

### 树哈希

$$
f_{now}=1+\sum{f_{son(now,i)}}\times prime(size(son(now,i)))
$$

其中 $f$ 表示哈希值， $prime(i)$ 表示第 $i$ 个质数，$son(x,i)$ 表示 $i$​ 的子节点之一，不需要排序。

### 2-SAT

每个位置拆成两个，$i$ 表示满足，$i'$ 表示不满足。 $i\to j$ 表示 $i$ 如果满足，那么 $j$ 一定要满足。

跑缩点，如果 $i$ 和 $i'$ 在一个强连通分量里面，那么说明不能满足。

构造解：如果要求字典序最小，对每个位置钦定选 $i/i'$ ，然后向后 dfs 判断是否会出现问题；如果不要求字典序最小，直接利用 Tarjan 的结果，若 $col[i]<col[i']$ ，则选择 $i$ （前提是使用 Tarjan 找环）。

## 数学相关

### 高斯消元

```cpp
void gauss(int n)
{
	for(int i=1;i<=n;++i)
	{
		int r=i;
		for(int j=i+1;j<=n;++j) if(fabs(a[r][i])<fabs(a[j][i])) r=j;
		if(r!=i) swap(a[i],a[r]);
		for(int j=n+1;j>=i;--j) a[i][j]/=a[i][i];
		for(int j=i+1;j<=n;++j)
			for(int k=n+1;k>=i;--k) a[j][k]-=a[j][i]*a[i][k];
	}
	ans[n]=a[n][n+1];
	for(int i=n-1;i;--i)
	{
		ans[i]=a[i][n+1];
		for(int j=i+1;j<=n;++j) ans[i]-=a[i][j]*ans[j];
	}
}
```

求行列式：

```cpp
int gauss()
{
	int ans=1;
	for(int i=1;i<n;i++)
	{
		if(!a[i][i])
			for(int j=i+1;j<n;j++) if(a[j][i])
			{
				ans=Mod-ans;
				swap(a[i],a[j]);
				break;
			}
		if (a[i][i]==0) exit(0);
		int inv=po(a[i][i],Mod-2);
		for(int j=i+1;j<n;j++)
		{
			int d=1ll*inv*a[j][i]%Mod;
			for(int k=i;k<n;k++)
				a[j][k]=(a[j][k]-1ll*d*a[i][k]%Mod+Mod)%Mod;
		}
		ans=1ll*ans*a[i][i]%Mod;
	}
	return ans;
}
```

### 线性基

应用：$k$ 大异或和

```cpp
void ins(ll a) //线性基核心部分
{
	for(int i=50;i>=0;i--)
		if((a>>i)&1ll) 
		{
			if(v[i]) a^=v[i];
			else
			{
				v[i]=a;
				return ;
			}
		}
	if(!a) flag=1;
	return;
}
void check() //execute after inserting all elements.
{
	for(int i=50;i>=0;i--)
		for(int j=i-1;j>=0;j--)
			if((v[i]>>j)&1) v[i]^=v[j];
	for(int i=50;i>=0;i--) cnt+=(v[i]>0);
}
ll query(ll k)
{
	if(k>(1ll<<cnt)) return -1;
	if(flag) k--;
	if(k==-1) return 0;
	ll res=0;
	for(int i=0;i<=50&&k;i++)
		if(v[i])
		{
			res^=(k&1)*v[i];
			k>>=1;
		}
	return res;
}
```

### 分治 NTT

```cpp
void init(int x)
{
	for(m=1;m<=x;m<<=1,++l);
	for(int i=0;i<m;++i) r[i]=(r[i>>1]>>1)|((i&1)<<(l-1));
}
#define mul(x,y) (1ll*x*y%Mod)
namespace poly
{
	int r[N],a[N],b[N],w[N],iw[N],m,l;
	void init(int x)
	{
		for(m=1,l=0;m<=x;m<<=1,++l);
		for(int i=0;i<m;++i) r[i]=(r[i>>1]>>1)|((i&1)<<(l-1));
		int b=po(3,(Mod-1)/m),ib=po(b,Mod-2);
		w[m/2]=iw[m/2]=1;
		for(int i=1;i<m/2;++i) w[m/2+i]=mul(w[m/2+i-1],b),iw[m/2+i]=mul(iw[m/2+i-1],ib);
		for(int i=m/2-1;i;--i) w[i]=w[i<<1],iw[i]=iw[i<<1];
	}
	void ntt(int* a, int f)
	{
		for(int i=0;i<m;++i) if(i<r[i]) swap(a[i],a[r[i]]);
		for(int i=1,id=1;i<m;i<<=1,++id)
		{
			for(int j=0;j<m;j+=i<<1)
				for(int k=0;k<i;++k)
				{
					int x=a[j+k],y=mul((f==1?w[i+k]:iw[i+k]),a[i+j+k]);
					a[j+k]=(x+y>=Mod?x+y-Mod:x+y);
					a[i+j+k]=(x-y<0?x-y+Mod:x-y);
				}
		}
		if(f==-1)
		{
			int in=po(m,Mod-2);
			for(int i=0;i<m;++i) a[i]=mul(a[i],in);
		}
	}
	void cdqntt(int l, int r)
	{
		if(l==r)
		{
			if(!l) f[l]=1;
			return ;
		}
		int mid=l+r>>1;
		cdqntt(l,mid);
		for(int i=l;i<=mid;++i) a[i-l]=f[i];
		for(int i=1;i<=r-l+1;++i) b[i]=g[i];
		if(r-l+1<=85)
		{
			for(int i=0;i<=mid-l;++i)
				for(int j=max(1,mid-i-l+1);j<=r-l+1&&i+j+l<=r;++j) upd(f[i+j+l],mul(a[i],b[j]));
		}
		else
		{
			init(r-l);
			memset(a+(mid-l+1),0,sizeof(int)*(m-(mid-l+1)));
			memset(b+(r-l+2),0,sizeof(m-(r-l+2)));
			ntt(a,1),ntt(b,1);
			for(int i=0;i<m;++i) a[i]=mul(a[i],b[i]);
			ntt(a,-1);
			for(int i=mid+1;i<=r;++i) upd(f[i],a[i-l]);
		}
		cdqntt(mid+1,r);
	}
}
```

### FWT

```cpp
const int Mod=1e9+7,iv2=Mod+1>>1;
#define mul(x,y) (1ll*(x)*(y)%Mod)
inline int add(int x, int y)
{	return (x+y>=Mod?x+y-Mod:x+y);
}
inline int sub(int x, int y)
{	return (x-y<0?x-y+Mod:x-y);
}
void FWT_or(int *a, int o)
{
	for(int i=1;i<n;i<<=1)
		for(int j=0;j<n;j+=i<<1)
			for(int k=0;k<i;++k)
				a[i+j+k]=(o==1?add(a[i+j+k],a[j+k]):sub(a[i+j+k],a[j+k]));
}
void FWT_and(int *a, int o)
{
	for(int i=1;i<n;i<<=1)
		for(int j=0;j<n;j+=i<<1)
			for(int k=0;k<i;++k)
				a[j+k]=(o==1?add(a[j+k],a[i+j+k]):sub(a[j+k],a[i+j+k]));
}
void FWT_xor(int *a, int o)
{
	for(int i=1;i<n;i<<=1)
		for(int j=0;j<n;j+=i<<1)
			for(int k=0;k<i;++k)
			{
				int x=a[j+k],y=a[i+j+k];
				a[j+k]=add(x,y),a[i+j+k]=sub(x,y);
				if(o==-1) a[j+k]=mul(a[j+k],iv2),a[i+j+k]=mul(a[i+j+k],iv2);
			}
}
```

### 拉格朗日插值

拉格朗日插值公式：$f(k)=\sum_{i=0}^ny_i\prod_{i\neq j} \frac{k-x_j}{x_i-x_j}$​；

如果得到 $n$ 个连续的点值，可以利用前缀积优化至 $O(n)$ .

```cpp
//求自然数幂和
int lagpow(ll n, int c)
{
	c+=2;
	inv[0]=inv[1]=1;
	for(int i=2;i<=c;++i) inv[i]=mul(Mod-Mod/i,inv[Mod%i]);
	for(int i=2;i<=c;++i) inv[i]=mul(inv[i],inv[i-1]);
	pre[0]=suf[c+1]=1;
	for(int i=1;i<=c;++i) pre[i]=mul(pre[i-1],(n-i)%Mod);
	for(int i=c;i>=1;--i) suf[i]=mul(suf[i+1],(n-i)%Mod);
	int sum=0,ans=0;
	for(int i=1;i<=c;++i)
	{
		sum=add(sum,po(i,c-2)); //sum 即为 y_i
		int t=mul(mul(inv[i-1],inv[c-i]),mul(pre[i-1],suf[i+1]));
		t=mul(t,sum);
		ans=((c-i&1)?sub(ans,t):add(ans,t));
	}
	return ans;
}
```

### BM算法求递推式+线性递推

```cpp
//by fjzzq2002
const int MOD=1e9+7;
ll qp(ll a,ll b)
{
    ll x=1; a%=MOD;
    for(;b;b>>=1,a=a*a%MOD) if(b&1) x=x*a%MOD;
    return x;
}
namespace linear_seq {
inline vector<int> BM(vector<int> x)
{
    vector<int> ls,cur;
    int pn=0,lf,ld;
    for(int i=0;i<int(x.size());++i)
    {
        ll t=-x[i]%MOD;
        for(int j=0;j<int(cur.size());++j)
            t=(t+x[i-j-1]*(ll)cur[j])%MOD;
        if(!t) continue;
        if(!cur.size())
        {cur.resize(i+1); lf=i; ld=t; continue;}
        ll k=-t*qp(ld,MOD-2)%MOD;
        vector<int> c(i-lf-1); c.push_back(-k);
        for(int j=0;j<int(ls.size());++j) c.push_back(ls[j]*k%MOD);
        if(c.size()<cur.size()) c.resize(cur.size());
        for(int j=0;j<int(cur.size());++j)
            c[j]=(c[j]+cur[j])%MOD;
        if(i-lf+(int)ls.size()>=(int)cur.size())
            ls=cur,lf=i,ld=t;
        cur=c;
    }
    vector<int>&o=cur;
    for(int i=0;i<int(o.size());++i)
        o[i]=(o[i]%MOD+MOD)%MOD;
    return o;
}
const int SZ=1e5+5;
int N; ll a[SZ],h[SZ],t_[SZ],s[SZ],t[SZ];
inline void mull(ll*p,ll*q)
{
    for(int i=0;i<N+N;++i) t_[i]=0;
    for(int i=0;i<N;++i) if(p[i])
        for(int j=0;j<N;++j)
            t_[i+j]=(t_[i+j]+p[i]*q[j])%MOD;
    for(int i=N+N-1;i>=N;--i) if(t_[i])
        for(int j=N-1;~j;--j)
            t_[i-j-1]=(t_[i-j-1]+t_[i]*h[j])%MOD;
    for(int i=0;i<N;++i) p[i]=t_[i];
}
inline ll calc(ll K)
{
    for(int i=N;~i;--i) s[i]=t[i]=0;
    s[0]=1; if(N!=1) t[1]=1; else t[0]=h[0];
    for(;K;mull(t,t),K>>=1) if(K&1) mull(s,t); ll su=0;
    for(int i=0;i<N;++i) su=(su+s[i]*a[i])%MOD;
    return (su%MOD+MOD)%MOD;
}
inline int gao(vector<int> x,ll n)
{
    if(n<int(x.size())) return x[n];
    vector<int> v=BM(x); N=v.size(); if(!N) return 0;
    for(int i=0;i<N;++i) h[i]=v[i],a[i]=x[i];
    return calc(n);
}
}
```

