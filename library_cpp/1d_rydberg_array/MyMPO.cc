#include "MyMPO.h"
#include <itensor/all.h>
using namespace itensor;


MPO
mpo_pxp(const SiteSet s, const double omega)
{
	int N = length(s);
    MPO H = MPO(s);

    // list of necessary operators (nb I use convention S are spin-matrices)
    // row and column index
    Index i_idx = Index(2);
    Index j_idx = Index(2);    

    // projectors (1+2*Sz)/2
    ITensor P = ITensor(i_idx,j_idx);
    P.set(i_idx(1),j_idx(1),1.);

    // 2*Sx  
    ITensor X = ITensor(i_idx,j_idx);
    X.set(i_idx(1),j_idx(2),1.);
    X.set(i_idx(2),j_idx(1),1.);

    // Identity
    ITensor I = ITensor(i_idx,j_idx);
    I.set(i_idx(1),j_idx(1),1.);
    I.set(i_idx(2),j_idx(2),1.);


    // build MPO
    // link indeces
    Index r_idx, l_idx;

    for(int j=1; j <= N; j++)
    {
        Index sj = s(j);
        Index sjp = prime(s(j));
        ITensor Hj;

        if(j==1)
        {
            r_idx = Index(2,tinyformat::format("l=%d,Link",j));
            Hj = ITensor(sj,sjp,r_idx);

            for(int l=1; l<= dim(sj); l++)
            {
                for(int k=1; k<= dim(sjp); k++)
                {
                    // set I to the first bond-index
                    Hj.set(sj(l),sjp(k),r_idx(1),omega*elt(I,i_idx(l),j_idx(k)));
                    // set P to the second bond-index
                    Hj.set(sj(l),sjp(k),r_idx(2),omega*elt(P,i_idx(l),j_idx(k)));
                }
            }

        }

        if(j==2)
        {
            l_idx = r_idx;
            r_idx = Index(3,tinyformat::format("l=%d,Link",j));
            Hj = ITensor(sj,sjp,l_idx,r_idx);

            for(int l=1; l<= dim(sj); l++)
            {
                for(int k=1; k<= dim(sjp); k++)
                {
                    // set Id to element (1,1)
                    Hj.set(sj(l),sjp(k),l_idx(1),r_idx(1),elt(I,i_idx(l),j_idx(k)));

                    // set P in elemennt (1,2)
                    Hj.set(sj(l),sjp(k),l_idx(1),r_idx(2),elt(P,i_idx(l),j_idx(k)));

                    // set X to element (2,3)
                    Hj.set(sj(l),sjp(k),l_idx(2),r_idx(3),elt(X,i_idx(l),j_idx(k)));
                }
            }
        }

        if(j==3)
        {
            l_idx = r_idx;
            r_idx = Index(4,tinyformat::format("l=%d,Link",j));

            Hj = ITensor(sj,sjp,l_idx,r_idx);

            for(int l=1; l<= dim(sj); l++)
            {
                for(int k=1; k<= dim(sjp); k++)
                {
                    // set Id to element (1,1)
                    Hj.set(sj(l),sjp(k),l_idx(1),r_idx(1),elt(I,i_idx(l),j_idx(k)));

                    // set P in elemennt (1,2) and (3,4)
                    Hj.set(sj(l),sjp(k),l_idx(1),r_idx(2),elt(P,i_idx(l),j_idx(k)));
                    Hj.set(sj(l),sjp(k),l_idx(3),r_idx(4),elt(P,i_idx(l),j_idx(k)));

                    // set X to element (2,3)
                    Hj.set(sj(l),sjp(k),l_idx(2),r_idx(3),elt(X,i_idx(l),j_idx(k)));
                }
            }
        }

        if(j >=4 && j <= N-3)
        {
            l_idx = r_idx;
            r_idx = Index(4,tinyformat::format("l=%d,Link",j));

            Hj = ITensor(sj,sjp,l_idx,r_idx);

            for(int l=1; l<= dim(sj); l++)
            {
                for(int k=1; k<= dim(sjp); k++)
                {
                    // set Id to element (1,1) and (4,4)
                    Hj.set(sj(l),sjp(k),l_idx(1),r_idx(1),elt(I,i_idx(l),j_idx(k)));
                    Hj.set(sj(l),sjp(k),l_idx(4),r_idx(4),elt(I,i_idx(l),j_idx(k)));

                    // set P in elemennt (1,2) and (3,4)
                    Hj.set(sj(l),sjp(k),l_idx(1),r_idx(2),elt(P,i_idx(l),j_idx(k)));
                    Hj.set(sj(l),sjp(k),l_idx(3),r_idx(4),elt(P,i_idx(l),j_idx(k)));

                    // set X to element (2,3)
                    Hj.set(sj(l),sjp(k),l_idx(2),r_idx(3),elt(X,i_idx(l),j_idx(k)));
                }
            }
        }

        if(j==N-2)
        {
            l_idx = r_idx;
            r_idx = Index(4,tinyformat::format("l=%d,Link",j));

            Hj = ITensor(sj,sjp,l_idx,r_idx);

            for(int l=1; l<= dim(sj); l++)
            {
                for(int k=1; k<= dim(sjp); k++)
                {
                    // set Id to element (4,4)
                    Hj.set(sj(l),sjp(k),l_idx(4),r_idx(4),elt(I,i_idx(l),j_idx(k)));

                    // set P in elemennt (1,2) and (3,4)
                    Hj.set(sj(l),sjp(k),l_idx(1),r_idx(2),elt(P,i_idx(l),j_idx(k)));
                    Hj.set(sj(l),sjp(k),l_idx(3),r_idx(4),elt(P,i_idx(l),j_idx(k)));

                    // set X to element (2,3)
                    Hj.set(sj(l),sjp(k),l_idx(2),r_idx(3),elt(X,i_idx(l),j_idx(k)));
                }
            }
        }

        if(j==N-1)
        {
            l_idx = r_idx;
            r_idx = Index(4,tinyformat::format("l=%d,Link",j));

            Hj = ITensor(sj,sjp,l_idx,r_idx);

            for(int l=1; l<= dim(sj); l++)
            {
                for(int k=1; k<= dim(sjp); k++)
                {
                    // set Id to element (4,4)
                    Hj.set(sj(l),sjp(k),l_idx(4),r_idx(4),elt(I,i_idx(l),j_idx(k)));

                    // set P in elemennt (3,4)
                    Hj.set(sj(l),sjp(k),l_idx(3),r_idx(4),elt(P,i_idx(l),j_idx(k)));

                    // set X to element (2,3)
                    Hj.set(sj(l),sjp(k),l_idx(2),r_idx(3),elt(X,i_idx(l),j_idx(k)));
                }
            }
        }

        if(j==N)
        {
            l_idx = r_idx;

            Hj = ITensor(sj,sjp,l_idx);

            for(int l=1; l<= dim(sj); l++)
            {
                for(int k=1; k<= dim(sjp); k++)
                {
                    // set Id to element (4,)
                    Hj.set(sj(l),sjp(k),l_idx(4),elt(I,i_idx(l),j_idx(k)));

                    // set P in elemennt (3,)
                    Hj.set(sj(l),sjp(k),l_idx(3),elt(P,i_idx(l),j_idx(k)));
                }
            }
        }


        H.ref(j) = Hj;
        
    }

    
    return H;

}
