#include<bits/stdc++.h>
using namespace std;
int main()
{
	freopen("template.md","r",stdin);
	freopen("body.tex","w",stdout);
	string s;
	while(getline(cin,s))
	{
		if(s.size()>=3&&s[0]=='#'&&s[1]=='#'&&s[2]=='#')
		{
			cout<<"\\subsection{";
			for(int i=4;i<s.size();++i) cout<<s[i];
			cout<<"}"<<endl;
		}
		else if(s.size()>=2&&s[0]=='#'&&s[1]=='#')
		{
			cout<<"\\section{";
			for(int i=3;i<s.size();++i) cout<<s[i];
			cout<<"}"<<endl;
		}
		else if(s=="```cpp")
			cout<<"\\begin{lstlisting}[language={C++}]"<<endl;
		else if(s=="```")
			cout<<"\\end{lstlisting}"<<endl;
		else cout<<s<<endl;
	}
	freopen("con","w",stdout);
	system("xelatex main.tex");
}
