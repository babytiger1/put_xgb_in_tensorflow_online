import pandas as pd
import tensorflow as tf 


def XGBprocess(data):
    s= [] 
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<0.875352502 or data["var17"] ==0) and (data["var22"]<-2.16016388 or data["var22"] ==0)):
             s.append("015")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<0.875352502 or data["var17"] ==0) and (data["var22"]>=-2.16016388 and data["var22"] !=0) and (data["var40"]<2.91252041 or data["var40"] ==0) and (data["var34"]<-2.12429523 or data["var34"] ==0)):
             s.append("055")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<0.875352502 or data["var17"] ==0) and (data["var22"]>=-2.16016388 and data["var22"] !=0) and (data["var40"]<2.91252041 or data["var40"] ==0) and (data["var34"]>=-2.12429523 and data["var34"] !=0) and (data["var48"]<-2.14176345 or data["var48"] ==0)):
             s.append("091")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<0.875352502 or data["var17"] ==0) and (data["var22"]>=-2.16016388 and data["var22"] !=0) and (data["var40"]<2.91252041 or data["var40"] ==0) and (data["var34"]>=-2.12429523 and data["var34"] !=0) and (data["var48"]>=-2.14176345 and data["var48"] !=0)):
             s.append("092")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<0.875352502 or data["var17"] ==0) and (data["var22"]>=-2.16016388 and data["var22"] !=0) and (data["var40"]>=2.91252041 and data["var40"] !=0)):
             s.append("032")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]>=0.875352502 and data["var17"] !=0) and (data["var21"]<0.520791292 or data["var21"] ==0) and (data["var36"]<0.256643981 or data["var36"] ==0) and (data["var11"]<0.986605406 or data["var11"] ==0)):
             s.append("057")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]>=0.875352502 and data["var17"] !=0) and (data["var21"]<0.520791292 or data["var21"] ==0) and (data["var36"]<0.256643981 or data["var36"] ==0) and (data["var11"]>=0.986605406 and data["var11"] !=0)):
             s.append("058")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]>=0.875352502 and data["var17"] !=0) and (data["var21"]<0.520791292 or data["var21"] ==0) and (data["var36"]>=0.256643981 and data["var36"] !=0)):
             s.append("034")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]>=0.875352502 and data["var17"] !=0) and (data["var21"]>=0.520791292 and data["var21"] !=0)):
             s.append("018")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]<-1.24065614 or data["var21"] ==0) and (data["var36"]<1.22742963 or data["var36"] ==0) and (data["var25"]<-2.37223125 or data["var25"] ==0)):
             s.append("059")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]<-1.24065614 or data["var21"] ==0) and (data["var36"]<1.22742963 or data["var36"] ==0) and (data["var25"]>=-2.37223125 and data["var25"] !=0) and (data["var19"]<-2.07533741 or data["var19"] ==0)):
             s.append("093")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]<-1.24065614 or data["var21"] ==0) and (data["var36"]<1.22742963 or data["var36"] ==0) and (data["var25"]>=-2.37223125 and data["var25"] !=0) and (data["var19"]>=-2.07533741 and data["var19"] !=0)):
             s.append("094")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]<-1.24065614 or data["var21"] ==0) and (data["var36"]>=1.22742963 and data["var36"] !=0) and (data["var27"]<0.409669131 or data["var27"] ==0)):
             s.append("061")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]<-1.24065614 or data["var21"] ==0) and (data["var36"]>=1.22742963 and data["var36"] !=0) and (data["var27"]>=0.409669131 and data["var27"] !=0)):
             s.append("062")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]<0.734828055 or data["var50"] ==0) and (data["var55"]<0.436708987 or data["var55"] ==0) and (data["var57"]<-1.62303495 or data["var57"] ==0)):
             s.append("095")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]<0.734828055 or data["var50"] ==0) and (data["var55"]<0.436708987 or data["var55"] ==0) and (data["var57"]>=-1.62303495 and data["var57"] !=0)):
             s.append("096")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]<0.734828055 or data["var50"] ==0) and (data["var55"]>=0.436708987 and data["var55"] !=0) and (data["var43"]<1.17869151 or data["var43"] ==0)):
             s.append("097")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]<0.734828055 or data["var50"] ==0) and (data["var55"]>=0.436708987 and data["var55"] !=0) and (data["var43"]>=1.17869151 and data["var43"] !=0)):
             s.append("098")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]>=0.734828055 and data["var50"] !=0) and (data["var54"]<-0.966283083 or data["var54"] ==0) and (data["var37"]<-0.452046633 or data["var37"] ==0)):
             s.append("099")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]>=0.734828055 and data["var50"] !=0) and (data["var54"]<-0.966283083 or data["var54"] ==0) and (data["var37"]>=-0.452046633 and data["var37"] !=0)):
             s.append("0100")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]>=0.734828055 and data["var50"] !=0) and (data["var54"]>=-0.966283083 and data["var54"] !=0) and (data["var57"]<0.913910985 or data["var57"] ==0)):
             s.append("0101")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]<1.93426228 or data["var30"] ==0) and (data["var21"]>=-1.24065614 and data["var21"] !=0) and (data["var50"]>=0.734828055 and data["var50"] !=0) and (data["var54"]>=-0.966283083 and data["var54"] !=0) and (data["var57"]>=0.913910985 and data["var57"] !=0)):
             s.append("0102")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]>=1.93426228 and data["var30"] !=0) and (data["var36"]<-2.11946559 or data["var36"] ==0)):
             s.append("021")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]>=1.93426228 and data["var30"] !=0) and (data["var36"]>=-2.11946559 and data["var36"] !=0) and (data["var01"]<-2.18531823 or data["var01"] ==0)):
             s.append("039")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var30"]>=1.93426228 and data["var30"] !=0) and (data["var36"]>=-2.11946559 and data["var36"] !=0) and (data["var01"]>=-2.18531823 and data["var01"] !=0)):
             s.append("040")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-0.762669444 or data["var27"] ==0) and (data["var13"]<2.32689619 or data["var13"] ==0) and (data["var40"]<2.087677 or data["var40"] ==0) and (data["var21"]<-2.77495432 or data["var21"] ==0)):
             s.append("067")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-0.762669444 or data["var27"] ==0) and (data["var13"]<2.32689619 or data["var13"] ==0) and (data["var40"]<2.087677 or data["var40"] ==0) and (data["var21"]>=-2.77495432 and data["var21"] !=0) and (data["var28"]<2.53138065 or data["var28"] ==0)):
             s.append("0103")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-0.762669444 or data["var27"] ==0) and (data["var13"]<2.32689619 or data["var13"] ==0) and (data["var40"]<2.087677 or data["var40"] ==0) and (data["var21"]>=-2.77495432 and data["var21"] !=0) and (data["var28"]>=2.53138065 and data["var28"] !=0)):
             s.append("0104")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-0.762669444 or data["var27"] ==0) and (data["var13"]<2.32689619 or data["var13"] ==0) and (data["var40"]>=2.087677 and data["var40"] !=0)):
             s.append("042")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-0.762669444 or data["var27"] ==0) and (data["var13"]>=2.32689619 and data["var13"] !=0)):
             s.append("024")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]<0.349930704 or data["var07"] ==0) and (data["var07"]<0.00514501799 or data["var07"] ==0) and (data["var08"]<-0.996021032 or data["var08"] ==0) and (data["var58"]<-1.08892322 or data["var58"] ==0)):
             s.append("0105")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]<0.349930704 or data["var07"] ==0) and (data["var07"]<0.00514501799 or data["var07"] ==0) and (data["var08"]<-0.996021032 or data["var08"] ==0) and (data["var58"]>=-1.08892322 and data["var58"] !=0)):
             s.append("0106")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]<0.349930704 or data["var07"] ==0) and (data["var07"]<0.00514501799 or data["var07"] ==0) and (data["var08"]>=-0.996021032 and data["var08"] !=0) and (data["var32"]<-0.751752734 or data["var32"] ==0)):
             s.append("0107")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]<0.349930704 or data["var07"] ==0) and (data["var07"]<0.00514501799 or data["var07"] ==0) and (data["var08"]>=-0.996021032 and data["var08"] !=0) and (data["var32"]>=-0.751752734 and data["var32"] !=0)):
             s.append("0108")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]<0.349930704 or data["var07"] ==0) and (data["var07"]>=0.00514501799 and data["var07"] !=0) and (data["var10"]<-1.14291286 or data["var10"] ==0)):
             s.append("071")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]<0.349930704 or data["var07"] ==0) and (data["var07"]>=0.00514501799 and data["var07"] !=0) and (data["var10"]>=-1.14291286 and data["var10"] !=0)):
             s.append("072")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]>=0.349930704 and data["var07"] !=0) and (data["var09"]<-1.36478424 or data["var09"] ==0) and (data["var10"]<0.307047158 or data["var10"] ==0)):
             s.append("073")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]>=0.349930704 and data["var07"] !=0) and (data["var09"]<-1.36478424 or data["var09"] ==0) and (data["var10"]>=0.307047158 and data["var10"] !=0) and (data["var04"]<-0.751992106 or data["var04"] ==0)):
             s.append("0109")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]>=0.349930704 and data["var07"] !=0) and (data["var09"]<-1.36478424 or data["var09"] ==0) and (data["var10"]>=0.307047158 and data["var10"] !=0) and (data["var04"]>=-0.751992106 and data["var04"] !=0)):
             s.append("0110")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]>=0.349930704 and data["var07"] !=0) and (data["var09"]>=-1.36478424 and data["var09"] !=0) and (data["var33"]<2.20350504 or data["var33"] ==0) and (data["var49"]<1.32830381 or data["var49"] ==0)):
             s.append("0111")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]>=0.349930704 and data["var07"] !=0) and (data["var09"]>=-1.36478424 and data["var09"] !=0) and (data["var33"]<2.20350504 or data["var33"] ==0) and (data["var49"]>=1.32830381 and data["var49"] !=0)):
             s.append("0112")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]>=0.349930704 and data["var07"] !=0) and (data["var09"]>=-1.36478424 and data["var09"] !=0) and (data["var33"]>=2.20350504 and data["var33"] !=0) and (data["var12"]<0.0812864751 or data["var12"] ==0)):
             s.append("0113")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-0.762669444 and data["var27"] !=0) and (data["var07"]>=0.349930704 and data["var07"] !=0) and (data["var09"]>=-1.36478424 and data["var09"] !=0) and (data["var33"]>=2.20350504 and data["var33"] !=0) and (data["var12"]>=0.0812864751 and data["var12"] !=0)):
             s.append("0114")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var10"]<0.590768337 or data["var10"] ==0)):
             s.append("0115")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var10"]>=0.590768337 and data["var10"] !=0)):
             s.append("0116")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]>=0.335894614 and data["var29"] !=0) and (data["var10"]<-1.96327543 or data["var10"] ==0)):
             s.append("0117")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]>=0.335894614 and data["var29"] !=0) and (data["var10"]>=-1.96327543 and data["var10"] !=0)):
             s.append("0118")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var08"]<-2.18796206 or data["var08"] ==0) and (data["var09"]<0.103131451 or data["var09"] ==0)):
             s.append("0119")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var08"]<-2.18796206 or data["var08"] ==0) and (data["var09"]>=0.103131451 and data["var09"] !=0)):
             s.append("0120")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var08"]>=-2.18796206 and data["var08"] !=0) and (data["var02"]<-1.29424691 or data["var02"] ==0)):
             s.append("0121")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]<-1.67283797 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var08"]>=-2.18796206 and data["var08"] !=0) and (data["var02"]>=-1.29424691 and data["var02"] !=0)):
             s.append("0122")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]<2.38949537 or data["var23"] ==0) and (data["var58"]<2.4114542 or data["var58"] ==0) and (data["var55"]<2.24234605 or data["var55"] ==0)):
             s.append("0123")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]<2.38949537 or data["var23"] ==0) and (data["var58"]<2.4114542 or data["var58"] ==0) and (data["var55"]>=2.24234605 and data["var55"] !=0)):
             s.append("0124")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]<2.38949537 or data["var23"] ==0) and (data["var58"]>=2.4114542 and data["var58"] !=0) and (data["var15"]<0.182395443 or data["var15"] ==0)):
             s.append("0125")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]<2.38949537 or data["var23"] ==0) and (data["var58"]>=2.4114542 and data["var58"] !=0) and (data["var15"]>=0.182395443 and data["var15"] !=0)):
             s.append("0126")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]>=2.38949537 and data["var23"] !=0) and (data["var23"]<3.4311161 or data["var23"] ==0) and (data["var02"]<0.82996726 or data["var02"] ==0)):
             s.append("0127")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]>=2.38949537 and data["var23"] !=0) and (data["var23"]<3.4311161 or data["var23"] ==0) and (data["var02"]>=0.82996726 and data["var02"] !=0)):
             s.append("0128")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]>=2.38949537 and data["var23"] !=0) and (data["var23"]>=3.4311161 and data["var23"] !=0) and (data["var26"]<2.41172004 or data["var26"] ==0)):
             s.append("0129")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.3934474 or data["var10"] ==0) and (data["var19"]>=-1.67283797 and data["var19"] !=0) and (data["var23"]>=2.38949537 and data["var23"] !=0) and (data["var23"]>=3.4311161 and data["var23"] !=0) and (data["var26"]>=2.41172004 and data["var26"] !=0)):
             s.append("0130")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]<-0.46432656 or data["var36"] ==0) and (data["var11"]<0.484783292 or data["var11"] ==0) and (data["var48"]<0.112738118 or data["var48"] ==0)):
             s.append("0131")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]<-0.46432656 or data["var36"] ==0) and (data["var11"]<0.484783292 or data["var11"] ==0) and (data["var48"]>=0.112738118 and data["var48"] !=0)):
             s.append("0132")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]<-0.46432656 or data["var36"] ==0) and (data["var11"]>=0.484783292 and data["var11"] !=0) and (data["var29"]<-1.0830369 or data["var29"] ==0)):
             s.append("0133")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]<-0.46432656 or data["var36"] ==0) and (data["var11"]>=0.484783292 and data["var11"] !=0) and (data["var29"]>=-1.0830369 and data["var29"] !=0)):
             s.append("0134")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]>=-0.46432656 and data["var36"] !=0) and (data["var31"]<1.38491821 or data["var31"] ==0) and (data["var06"]<-1.7955395 or data["var06"] ==0)):
             s.append("0135")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]>=-0.46432656 and data["var36"] !=0) and (data["var31"]<1.38491821 or data["var31"] ==0) and (data["var06"]>=-1.7955395 and data["var06"] !=0)):
             s.append("0136")   
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]>=-0.46432656 and data["var36"] !=0) and (data["var31"]>=1.38491821 and data["var31"] !=0) and (data["var50"]<-0.375242054 or data["var50"] ==0)):
             s.append("0137")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]>=-0.46432656 and data["var36"] !=0) and (data["var31"]>=1.38491821 and data["var31"] !=0) and (data["var50"]>=-0.375242054 and data["var50"] !=0)):
             s.append("0138")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]>=3.56514144 and data["var10"] !=0) and (data["var05"]<-1.75495124 or data["var05"] ==0)):
             s.append("053")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]>=3.56514144 and data["var10"] !=0) and (data["var05"]>=-1.75495124 and data["var05"] !=0) and (data["var02"]<-1.57104671 or data["var02"] ==0)):
             s.append("089")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]>=3.56514144 and data["var10"] !=0) and (data["var05"]>=-1.75495124 and data["var05"] !=0) and (data["var02"]>=-1.57104671 and data["var02"] !=0) and (data["var11"]<-1.6163739 or data["var11"] ==0)):
             s.append("0139")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.3934474 and data["var10"] !=0) and (data["var10"]>=3.56514144 and data["var10"] !=0) and (data["var05"]>=-1.75495124 and data["var05"] !=0) and (data["var02"]>=-1.57104671 and data["var02"] !=0) and (data["var11"]>=-1.6163739 and data["var11"] !=0)):
             s.append("0140")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]<-1.61443472 or data["var44"] ==0) and (data["var27"]<-1.49087071 or data["var27"] ==0) and (data["var29"]<0.202808917 or data["var29"] ==0)):
             s.append("115")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]<-1.61443472 or data["var44"] ==0) and (data["var27"]<-1.49087071 or data["var27"] ==0) and (data["var29"]>=0.202808917 and data["var29"] !=0)):
             s.append("116")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]<-1.61443472 or data["var44"] ==0) and (data["var27"]>=-1.49087071 and data["var27"] !=0) and (data["var12"]<1.75823832 or data["var12"] ==0) and (data["var26"]<1.49490523 or data["var26"] ==0)):
             s.append("131")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]<-1.61443472 or data["var44"] ==0) and (data["var27"]>=-1.49087071 and data["var27"] !=0) and (data["var12"]<1.75823832 or data["var12"] ==0) and (data["var26"]>=1.49490523 and data["var26"] !=0)):
             s.append("132")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]<-1.61443472 or data["var44"] ==0) and (data["var27"]>=-1.49087071 and data["var27"] !=0) and (data["var12"]>=1.75823832 and data["var12"] !=0)):
             s.append("118")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]<0.892111659 or data["var17"] ==0) and (data["var22"]<-2.16016388 or data["var22"] ==0)):
             s.append("133")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]<0.892111659 or data["var17"] ==0) and (data["var22"]>=-2.16016388 and data["var22"] !=0) and (data["var53"]<1.24496448 or data["var53"] ==0)):
             s.append("157")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]<0.892111659 or data["var17"] ==0) and (data["var22"]>=-2.16016388 and data["var22"] !=0) and (data["var53"]>=1.24496448 and data["var53"] !=0) and (data["var11"]<-0.309529632 or data["var11"] ==0)):
             s.append("195")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]<0.892111659 or data["var17"] ==0) and (data["var22"]>=-2.16016388 and data["var22"] !=0) and (data["var53"]>=1.24496448 and data["var53"] !=0) and (data["var11"]>=-0.309529632 and data["var11"] !=0)):
             s.append("196")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]>=0.892111659 and data["var17"] !=0) and (data["var53"]<-0.236049742 or data["var53"] ==0) and (data["var06"]<0.757528663 or data["var06"] ==0)):
             s.append("159")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]>=0.892111659 and data["var17"] !=0) and (data["var53"]<-0.236049742 or data["var53"] ==0) and (data["var06"]>=0.757528663 and data["var06"] !=0)):
             s.append("160")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]>=0.892111659 and data["var17"] !=0) and (data["var53"]>=-0.236049742 and data["var53"] !=0) and (data["var02"]<-0.170746699 or data["var02"] ==0)):
             s.append("161")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]<-1.14670086 or data["var42"] ==0) and (data["var17"]>=0.892111659 and data["var17"] !=0) and (data["var53"]>=-0.236049742 and data["var53"] !=0) and (data["var02"]>=-0.170746699 and data["var02"] !=0)):
             s.append("162")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]<0.350182056 or data["var44"] ==0) and (data["var13"]<-1.73488581 or data["var13"] ==0) and (data["var56"]<-1.64179909 or data["var56"] ==0)):
             s.append("163")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]<0.350182056 or data["var44"] ==0) and (data["var13"]<-1.73488581 or data["var13"] ==0) and (data["var56"]>=-1.64179909 and data["var56"] !=0)):
             s.append("164")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]<0.350182056 or data["var44"] ==0) and (data["var13"]>=-1.73488581 and data["var13"] !=0) and (data["var25"]<1.92558861 or data["var25"] ==0) and (data["var58"]<-0.741109848 or data["var58"] ==0)):
             s.append("197")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]<0.350182056 or data["var44"] ==0) and (data["var13"]>=-1.73488581 and data["var13"] !=0) and (data["var25"]<1.92558861 or data["var25"] ==0) and (data["var58"]>=-0.741109848 and data["var58"] !=0)):
             s.append("198")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]<0.350182056 or data["var44"] ==0) and (data["var13"]>=-1.73488581 and data["var13"] !=0) and (data["var25"]>=1.92558861 and data["var25"] !=0) and (data["var30"]<0.311072469 or data["var30"] ==0)):
             s.append("199")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]<0.350182056 or data["var44"] ==0) and (data["var13"]>=-1.73488581 and data["var13"] !=0) and (data["var25"]>=1.92558861 and data["var25"] !=0) and (data["var30"]>=0.311072469 and data["var30"] !=0)):
             s.append("1100")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]>=0.350182056 and data["var44"] !=0) and (data["var12"]<1.13213646 or data["var12"] ==0) and (data["var16"]<-1.51059246 or data["var16"] ==0)):
             s.append("167")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]>=0.350182056 and data["var44"] !=0) and (data["var12"]<1.13213646 or data["var12"] ==0) and (data["var16"]>=-1.51059246 and data["var16"] !=0) and (data["var37"]<-1.05360317 or data["var37"] ==0)):
             s.append("1101")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]>=0.350182056 and data["var44"] !=0) and (data["var12"]<1.13213646 or data["var12"] ==0) and (data["var16"]>=-1.51059246 and data["var16"] !=0) and (data["var37"]>=-1.05360317 and data["var37"] !=0)):
             s.append("1102")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]>=0.350182056 and data["var44"] !=0) and (data["var12"]>=1.13213646 and data["var12"] !=0) and (data["var02"]<-1.17420053 or data["var02"] ==0)):
             s.append("169")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var44"]>=-1.61443472 and data["var44"] !=0) and (data["var42"]>=-1.14670086 and data["var42"] !=0) and (data["var44"]>=0.350182056 and data["var44"] !=0) and (data["var12"]>=1.13213646 and data["var12"] !=0) and (data["var02"]>=-1.17420053 and data["var02"] !=0)):
             s.append("170")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]<0.0362802446 or data["var16"] ==0) and (data["var09"]<-1.95977926 or data["var09"] ==0)):
             s.append("141")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]<0.0362802446 or data["var16"] ==0) and (data["var09"]>=-1.95977926 and data["var09"] !=0) and (data["var56"]<0.363874555 or data["var56"] ==0) and (data["var16"]<-1.03087735 or data["var16"] ==0)):
             s.append("1103")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]<0.0362802446 or data["var16"] ==0) and (data["var09"]>=-1.95977926 and data["var09"] !=0) and (data["var56"]<0.363874555 or data["var56"] ==0) and (data["var16"]>=-1.03087735 and data["var16"] !=0)):
             s.append("1104")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]<0.0362802446 or data["var16"] ==0) and (data["var09"]>=-1.95977926 and data["var09"] !=0) and (data["var56"]>=0.363874555 and data["var56"] !=0) and (data["var04"]<-0.80365175 or data["var04"] ==0)):
             s.append("1105")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]<0.0362802446 or data["var16"] ==0) and (data["var09"]>=-1.95977926 and data["var09"] !=0) and (data["var56"]>=0.363874555 and data["var56"] !=0) and (data["var04"]>=-0.80365175 and data["var04"] !=0)):
             s.append("1106")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]>=0.0362802446 and data["var16"] !=0) and (data["var42"]<-0.975684762 or data["var42"] ==0) and (data["var18"]<-1.91885126 or data["var18"] ==0)):
             s.append("173")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]>=0.0362802446 and data["var16"] !=0) and (data["var42"]<-0.975684762 or data["var42"] ==0) and (data["var18"]>=-1.91885126 and data["var18"] !=0) and (data["var23"]<1.8980056 or data["var23"] ==0)):
             s.append("1107")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]>=0.0362802446 and data["var16"] !=0) and (data["var42"]<-0.975684762 or data["var42"] ==0) and (data["var18"]>=-1.91885126 and data["var18"] !=0) and (data["var23"]>=1.8980056 and data["var23"] !=0)):
             s.append("1108")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]<-0.902789891 or data["var42"] ==0) and (data["var16"]>=0.0362802446 and data["var16"] !=0) and (data["var42"]>=-0.975684762 and data["var42"] !=0)):
             s.append("144")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]<-1.7437216 or data["var01"] ==0) and (data["var55"]<1.16773891 or data["var55"] ==0) and (data["var34"]<1.660568 or data["var34"] ==0)):
             s.append("175")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]<-1.7437216 or data["var01"] ==0) and (data["var55"]<1.16773891 or data["var55"] ==0) and (data["var34"]>=1.660568 and data["var34"] !=0)):
             s.append("176")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]<-1.7437216 or data["var01"] ==0) and (data["var55"]>=1.16773891 and data["var55"] !=0)):
             s.append("146")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]>=-1.7437216 and data["var01"] !=0) and (data["var19"]<0.818261504 or data["var19"] ==0) and (data["var37"]<-2.43811131 or data["var37"] ==0)):
             s.append("177")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]>=-1.7437216 and data["var01"] !=0) and (data["var19"]<0.818261504 or data["var19"] ==0) and (data["var37"]>=-2.43811131 and data["var37"] !=0) and (data["var05"]<-0.99552232 or data["var05"] ==0)):
             s.append("1109")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]>=-1.7437216 and data["var01"] !=0) and (data["var19"]<0.818261504 or data["var19"] ==0) and (data["var37"]>=-2.43811131 and data["var37"] !=0) and (data["var05"]>=-0.99552232 and data["var05"] !=0)):
             s.append("1110")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]>=-1.7437216 and data["var01"] !=0) and (data["var19"]>=0.818261504 and data["var19"] !=0) and (data["var02"]<-0.801064253 or data["var02"] ==0) and (data["var05"]<-0.475177348 or data["var05"] ==0)):
             s.append("1111")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]>=-1.7437216 and data["var01"] !=0) and (data["var19"]>=0.818261504 and data["var19"] !=0) and (data["var02"]<-0.801064253 or data["var02"] ==0) and (data["var05"]>=-0.475177348 and data["var05"] !=0)):
             s.append("1112")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]>=-1.7437216 and data["var01"] !=0) and (data["var19"]>=0.818261504 and data["var19"] !=0) and (data["var02"]>=-0.801064253 and data["var02"] !=0) and (data["var53"]<1.30915713 or data["var53"] ==0)):
             s.append("1113")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]<-2.12194443 or data["var60"] ==0) and (data["var42"]>=-0.902789891 and data["var42"] !=0) and (data["var01"]>=-1.7437216 and data["var01"] !=0) and (data["var19"]>=0.818261504 and data["var19"] !=0) and (data["var02"]>=-0.801064253 and data["var02"] !=0) and (data["var53"]>=1.30915713 and data["var53"] !=0)):
             s.append("1114")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]<-2.08309984 or data["var58"] ==0) and (data["var07"]<0.675977468 or data["var07"] ==0) and (data["var23"]<-0.623989105 or data["var23"] ==0)):
             s.append("1115")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]<-2.08309984 or data["var58"] ==0) and (data["var07"]<0.675977468 or data["var07"] ==0) and (data["var23"]>=-0.623989105 and data["var23"] !=0)):
             s.append("1116")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]<-2.08309984 or data["var58"] ==0) and (data["var07"]>=0.675977468 and data["var07"] !=0) and (data["var04"]<2.36127281 or data["var04"] ==0)):
             s.append("1117")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]<-2.08309984 or data["var58"] ==0) and (data["var07"]>=0.675977468 and data["var07"] !=0) and (data["var04"]>=2.36127281 and data["var04"] !=0)):
             s.append("1118")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]>=-2.08309984 and data["var58"] !=0) and (data["var57"]<1.64801407 or data["var57"] ==0) and (data["var33"]<2.20363998 or data["var33"] ==0)):
             s.append("1119")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]>=-2.08309984 and data["var58"] !=0) and (data["var57"]<1.64801407 or data["var57"] ==0) and (data["var33"]>=2.20363998 and data["var33"] !=0)):
             s.append("1120")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]>=-2.08309984 and data["var58"] !=0) and (data["var57"]>=1.64801407 and data["var57"] !=0) and (data["var08"]<1.67481863 or data["var08"] ==0)):
             s.append("1121")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]<1.75158048 or data["var23"] ==0) and (data["var58"]>=-2.08309984 and data["var58"] !=0) and (data["var57"]>=1.64801407 and data["var57"] !=0) and (data["var08"]>=1.67481863 and data["var08"] !=0)):
             s.append("1122")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]<-1.48379517 or data["var09"] ==0) and (data["var17"]<-0.800522268 or data["var17"] ==0) and (data["var17"]<-1.08357728 or data["var17"] ==0)):
             s.append("1123")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]<-1.48379517 or data["var09"] ==0) and (data["var17"]<-0.800522268 or data["var17"] ==0) and (data["var17"]>=-1.08357728 and data["var17"] !=0)):
             s.append("1124")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]<-1.48379517 or data["var09"] ==0) and (data["var17"]>=-0.800522268 and data["var17"] !=0) and (data["var30"]<0.661472023 or data["var30"] ==0)):
             s.append("1125")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]<-1.48379517 or data["var09"] ==0) and (data["var17"]>=-0.800522268 and data["var17"] !=0) and (data["var30"]>=0.661472023 and data["var30"] !=0)):
             s.append("1126")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]>=-1.48379517 and data["var09"] !=0) and (data["var56"]<-1.5359695 or data["var56"] ==0) and (data["var39"]<-1.95140243 or data["var39"] ==0)):
             s.append("1127")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]>=-1.48379517 and data["var09"] !=0) and (data["var56"]<-1.5359695 or data["var56"] ==0) and (data["var39"]>=-1.95140243 and data["var39"] !=0)):
             s.append("1128")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]>=-1.48379517 and data["var09"] !=0) and (data["var56"]>=-1.5359695 and data["var56"] !=0) and (data["var53"]<1.30052733 or data["var53"] ==0)):
             s.append("1129")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]<2.87884068 or data["var39"] ==0) and (data["var23"]>=1.75158048 and data["var23"] !=0) and (data["var09"]>=-1.48379517 and data["var09"] !=0) and (data["var56"]>=-1.5359695 and data["var56"] !=0) and (data["var53"]>=1.30052733 and data["var53"] !=0)):
             s.append("1130")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.287269592 or data["var44"] ==0) and (data["var49"]<-1.23201203 or data["var49"] ==0)):
             s.append("153")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.287269592 or data["var44"] ==0) and (data["var49"]>=-1.23201203 and data["var49"] !=0) and (data["var13"]<-0.799175382 or data["var13"] ==0) and (data["var28"]<0.619602323 or data["var28"] ==0)):
             s.append("1131")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.287269592 or data["var44"] ==0) and (data["var49"]>=-1.23201203 and data["var49"] !=0) and (data["var13"]<-0.799175382 or data["var13"] ==0) and (data["var28"]>=0.619602323 and data["var28"] !=0)):
             s.append("1132")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.287269592 or data["var44"] ==0) and (data["var49"]>=-1.23201203 and data["var49"] !=0) and (data["var13"]>=-0.799175382 and data["var13"] !=0) and (data["var24"]<-0.515090346 or data["var24"] ==0)):
             s.append("1133")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.287269592 or data["var44"] ==0) and (data["var49"]>=-1.23201203 and data["var49"] !=0) and (data["var13"]>=-0.799175382 and data["var13"] !=0) and (data["var24"]>=-0.515090346 and data["var24"] !=0)):
             s.append("1134")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]<-1.18555641 or data["var41"] ==0) and (data["var37"]<0.448014796 or data["var37"] ==0) and (data["var16"]<0.583433628 or data["var16"] ==0)):
             s.append("1135")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]<-1.18555641 or data["var41"] ==0) and (data["var37"]<0.448014796 or data["var37"] ==0) and (data["var16"]>=0.583433628 and data["var16"] !=0)):
             s.append("1136")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]<-1.18555641 or data["var41"] ==0) and (data["var37"]>=0.448014796 and data["var37"] !=0) and (data["var42"]<0.777472258 or data["var42"] ==0)):
             s.append("1137")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]<-1.18555641 or data["var41"] ==0) and (data["var37"]>=0.448014796 and data["var37"] !=0) and (data["var42"]>=0.777472258 and data["var42"] !=0)):
             s.append("1138")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]>=-1.18555641 and data["var41"] !=0) and (data["var37"]<1.28700185 or data["var37"] ==0) and (data["var53"]<-1.96883726 or data["var53"] ==0)):
             s.append("1139")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]>=-1.18555641 and data["var41"] !=0) and (data["var37"]<1.28700185 or data["var37"] ==0) and (data["var53"]>=-1.96883726 and data["var53"] !=0)):
             s.append("1140")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]>=-1.18555641 and data["var41"] !=0) and (data["var37"]>=1.28700185 and data["var37"] !=0) and (data["var26"]<0.425676525 or data["var26"] ==0)):
             s.append("1141")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var60"]>=-2.12194443 and data["var60"] !=0) and (data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.287269592 and data["var44"] !=0) and (data["var41"]>=-1.18555641 and data["var41"] !=0) and (data["var37"]>=1.28700185 and data["var37"] !=0) and (data["var26"]>=0.425676525 and data["var26"] !=0)):
             s.append("1142")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]<-0.473163903 or data["var59"] ==0) and (data["var28"]<-2.14848852 or data["var28"] ==0)):
             s.append("229")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]<-0.473163903 or data["var59"] ==0) and (data["var28"]>=-2.14848852 and data["var28"] !=0) and (data["var24"]<-1.42699051 or data["var24"] ==0) and (data["var24"]<-1.74700725 or data["var24"] ==0)):
             s.append("273")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]<-0.473163903 or data["var59"] ==0) and (data["var28"]>=-2.14848852 and data["var28"] !=0) and (data["var24"]<-1.42699051 or data["var24"] ==0) and (data["var24"]>=-1.74700725 and data["var24"] !=0)):
             s.append("274")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]<-0.473163903 or data["var59"] ==0) and (data["var28"]>=-2.14848852 and data["var28"] !=0) and (data["var24"]>=-1.42699051 and data["var24"] !=0) and (data["var42"]<0.417148352 or data["var42"] ==0)):
             s.append("275")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]<-0.473163903 or data["var59"] ==0) and (data["var28"]>=-2.14848852 and data["var28"] !=0) and (data["var24"]>=-1.42699051 and data["var24"] !=0) and (data["var42"]>=0.417148352 and data["var42"] !=0)):
             s.append("276")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]<-0.593146205 or data["var50"] ==0) and (data["var39"]<-0.680132508 or data["var39"] ==0) and (data["var55"]<-0.710103154 or data["var55"] ==0)):
             s.append("277")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]<-0.593146205 or data["var50"] ==0) and (data["var39"]<-0.680132508 or data["var39"] ==0) and (data["var55"]>=-0.710103154 and data["var55"] !=0)):
             s.append("278")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]<-0.593146205 or data["var50"] ==0) and (data["var39"]>=-0.680132508 and data["var39"] !=0) and (data["var44"]<-1.78796744 or data["var44"] ==0)):
             s.append("279")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]<-0.593146205 or data["var50"] ==0) and (data["var39"]>=-0.680132508 and data["var39"] !=0) and (data["var44"]>=-1.78796744 and data["var44"] !=0)):
             s.append("280")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]>=-0.593146205 and data["var50"] !=0) and (data["var02"]<0.807540774 or data["var02"] ==0) and (data["var49"]<1.14276564 or data["var49"] ==0)):
             s.append("281")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]>=-0.593146205 and data["var50"] !=0) and (data["var02"]<0.807540774 or data["var02"] ==0) and (data["var49"]>=1.14276564 and data["var49"] !=0)):
             s.append("282")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]>=-0.593146205 and data["var50"] !=0) and (data["var02"]>=0.807540774 and data["var02"] !=0) and (data["var57"]<-0.241673797 or data["var57"] ==0)):
             s.append("283")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]<0.566927016 or data["var50"] ==0) and (data["var59"]>=-0.473163903 and data["var59"] !=0) and (data["var50"]>=-0.593146205 and data["var50"] !=0) and (data["var02"]>=0.807540774 and data["var02"] !=0) and (data["var57"]>=-0.241673797 and data["var57"] !=0)):
             s.append("284")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]>=0.566927016 and data["var50"] !=0) and (data["var58"]<1.30349183 or data["var58"] ==0) and (data["var34"]<1.54572499 or data["var34"] ==0) and (data["var07"]<1.59256434 or data["var07"] ==0) and (data["var06"]<1.84620309 or data["var06"] ==0)):
             s.append("285")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]>=0.566927016 and data["var50"] !=0) and (data["var58"]<1.30349183 or data["var58"] ==0) and (data["var34"]<1.54572499 or data["var34"] ==0) and (data["var07"]<1.59256434 or data["var07"] ==0) and (data["var06"]>=1.84620309 and data["var06"] !=0)):
             s.append("286")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]>=0.566927016 and data["var50"] !=0) and (data["var58"]<1.30349183 or data["var58"] ==0) and (data["var34"]<1.54572499 or data["var34"] ==0) and (data["var07"]>=1.59256434 and data["var07"] !=0) and (data["var40"]<-0.177815974 or data["var40"] ==0)):
             s.append("287")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]>=0.566927016 and data["var50"] !=0) and (data["var58"]<1.30349183 or data["var58"] ==0) and (data["var34"]<1.54572499 or data["var34"] ==0) and (data["var07"]>=1.59256434 and data["var07"] !=0) and (data["var40"]>=-0.177815974 and data["var40"] !=0)):
             s.append("288")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]>=0.566927016 and data["var50"] !=0) and (data["var58"]<1.30349183 or data["var58"] ==0) and (data["var34"]>=1.54572499 and data["var34"] !=0)):
             s.append("234")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]>=0.566927016 and data["var50"] !=0) and (data["var58"]>=1.30349183 and data["var58"] !=0) and (data["var06"]<-1.19075382 or data["var06"] ==0)):
             s.append("235")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]<1.70234013 or data["var57"] ==0) and (data["var50"]>=0.566927016 and data["var50"] !=0) and (data["var58"]>=1.30349183 and data["var58"] !=0) and (data["var06"]>=-1.19075382 and data["var06"] !=0)):
             s.append("236")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]>=1.70234013 and data["var57"] !=0) and (data["var33"]<-1.75797534 or data["var33"] ==0)):
             s.append("29")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]>=1.70234013 and data["var57"] !=0) and (data["var33"]>=-1.75797534 and data["var33"] !=0) and (data["var12"]<-1.34297609 or data["var12"] ==0)):
             s.append("219")
    if((data["var38"]<-2.49855733 or data["var38"] ==0) and (data["var57"]>=1.70234013 and data["var57"] !=0) and (data["var33"]>=-1.75797534 and data["var33"] !=0) and (data["var12"]>=-1.34297609 and data["var12"] !=0)):
             s.append("220")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]<-1.24643636 or data["var35"] ==0) and (data["var48"]<-1.36329412 or data["var48"] ==0)):
             s.append("221")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]<-1.24643636 or data["var35"] ==0) and (data["var48"]>=-1.36329412 and data["var48"] !=0) and (data["var26"]<2.19935369 or data["var26"] ==0)):
             s.append("237")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]<-1.24643636 or data["var35"] ==0) and (data["var48"]>=-1.36329412 and data["var48"] !=0) and (data["var26"]>=2.19935369 and data["var26"] !=0)):
             s.append("238")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]<-1.30538988 or data["var17"] ==0) and (data["var59"]<1.56881452 or data["var59"] ==0)):
             s.append("239")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]<-1.30538988 or data["var17"] ==0) and (data["var59"]>=1.56881452 and data["var59"] !=0)):
             s.append("240")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var44"]<1.2986697 or data["var44"] ==0) and (data["var06"]<-1.51368475 or data["var06"] ==0) and (data["var35"]<-0.147770658 or data["var35"] ==0)):
             s.append("289")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var44"]<1.2986697 or data["var44"] ==0) and (data["var06"]<-1.51368475 or data["var06"] ==0) and (data["var35"]>=-0.147770658 and data["var35"] !=0)):
             s.append("290")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var44"]<1.2986697 or data["var44"] ==0) and (data["var06"]>=-1.51368475 and data["var06"] !=0) and (data["var56"]<-0.713963866 or data["var56"] ==0)):
             s.append("291")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var44"]<1.2986697 or data["var44"] ==0) and (data["var06"]>=-1.51368475 and data["var06"] !=0) and (data["var56"]>=-0.713963866 and data["var56"] !=0)):
             s.append("292")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var44"]>=1.2986697 and data["var44"] !=0) and (data["var16"]<1.50797462 or data["var16"] ==0)):
             s.append("259")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var44"]>=1.2986697 and data["var44"] !=0) and (data["var16"]>=1.50797462 and data["var16"] !=0)):
             s.append("260")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]<-2.46717882 or data["var07"] ==0)):
             s.append("225")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var29"]<-2.10754991 or data["var29"] ==0) and (data["var59"]<1.60812902 or data["var59"] ==0) and (data["var56"]<-1.51848459 or data["var56"] ==0)):
             s.append("293")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var29"]<-2.10754991 or data["var29"] ==0) and (data["var59"]<1.60812902 or data["var59"] ==0) and (data["var56"]>=-1.51848459 and data["var56"] !=0)):
             s.append("294")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var29"]<-2.10754991 or data["var29"] ==0) and (data["var59"]>=1.60812902 and data["var59"] !=0)):
             s.append("262")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var29"]>=-2.10754991 and data["var29"] !=0) and (data["var30"]<1.3566587 or data["var30"] ==0) and (data["var58"]<2.37855887 or data["var58"] ==0)):
             s.append("295")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var29"]>=-2.10754991 and data["var29"] !=0) and (data["var30"]<1.3566587 or data["var30"] ==0) and (data["var58"]>=2.37855887 and data["var58"] !=0)):
             s.append("296")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var29"]>=-2.10754991 and data["var29"] !=0) and (data["var30"]>=1.3566587 and data["var30"] !=0) and (data["var34"]<0.700020432 or data["var34"] ==0)):
             s.append("297")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]<-2.03693771 or data["var04"] ==0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var29"]>=-2.10754991 and data["var29"] !=0) and (data["var30"]>=1.3566587 and data["var30"] !=0) and (data["var34"]>=0.700020432 and data["var34"] !=0)):
             s.append("298")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]<1.81177855 or data["var04"] ==0) and (data["var44"]<-2.7849679 or data["var44"] ==0) and (data["var01"]<-0.225437969 or data["var01"] ==0)):
             s.append("299")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]<1.81177855 or data["var04"] ==0) and (data["var44"]<-2.7849679 or data["var44"] ==0) and (data["var01"]>=-0.225437969 and data["var01"] !=0)):
             s.append("2100")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]<1.81177855 or data["var04"] ==0) and (data["var44"]>=-2.7849679 and data["var44"] !=0) and (data["var10"]<2.38673496 or data["var10"] ==0)):
             s.append("2101")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]<1.81177855 or data["var04"] ==0) and (data["var44"]>=-2.7849679 and data["var44"] !=0) and (data["var10"]>=2.38673496 and data["var10"] !=0)):
             s.append("2102")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]>=1.81177855 and data["var04"] !=0) and (data["var40"]<1.89110267 or data["var40"] ==0) and (data["var08"]<1.73919368 or data["var08"] ==0)):
             s.append("2103")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]>=1.81177855 and data["var04"] !=0) and (data["var40"]<1.89110267 or data["var40"] ==0) and (data["var08"]>=1.73919368 and data["var08"] !=0)):
             s.append("2104")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]>=1.81177855 and data["var04"] !=0) and (data["var40"]>=1.89110267 and data["var40"] !=0) and (data["var24"]<-0.768454671 or data["var24"] ==0)):
             s.append("2105")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]<2.25967646 or data["var15"] ==0) and (data["var04"]>=1.81177855 and data["var04"] !=0) and (data["var40"]>=1.89110267 and data["var40"] !=0) and (data["var24"]>=-0.768454671 and data["var24"] !=0)):
             s.append("2106")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]>=2.25967646 and data["var15"] !=0) and (data["var17"]<-1.45010662 or data["var17"] ==0) and (data["var27"]<0.791213572 or data["var27"] ==0)):
             s.append("269")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]>=2.25967646 and data["var15"] !=0) and (data["var17"]<-1.45010662 or data["var17"] ==0) and (data["var27"]>=0.791213572 and data["var27"] !=0) and (data["var55"]<0.21598798 or data["var55"] ==0)):
             s.append("2107")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]>=2.25967646 and data["var15"] !=0) and (data["var17"]<-1.45010662 or data["var17"] ==0) and (data["var27"]>=0.791213572 and data["var27"] !=0) and (data["var55"]>=0.21598798 and data["var55"] !=0)):
             s.append("2108")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]>=2.25967646 and data["var15"] !=0) and (data["var17"]>=-1.45010662 and data["var17"] !=0) and (data["var56"]<-0.105358839 or data["var56"] ==0) and (data["var11"]<1.10187066 or data["var11"] ==0)):
             s.append("2109")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]>=2.25967646 and data["var15"] !=0) and (data["var17"]>=-1.45010662 and data["var17"] !=0) and (data["var56"]<-0.105358839 or data["var56"] ==0) and (data["var11"]>=1.10187066 and data["var11"] !=0)):
             s.append("2110")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]>=2.25967646 and data["var15"] !=0) and (data["var17"]>=-1.45010662 and data["var17"] !=0) and (data["var56"]>=-0.105358839 and data["var56"] !=0) and (data["var56"]<1.08235955 or data["var56"] ==0)):
             s.append("2111")
    if((data["var38"]>=-2.49855733 and data["var38"] !=0) and (data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var04"]>=-2.03693771 and data["var04"] !=0) and (data["var15"]>=2.25967646 and data["var15"] !=0) and (data["var17"]>=-1.45010662 and data["var17"] !=0) and (data["var56"]>=-0.105358839 and data["var56"] !=0) and (data["var56"]>=1.08235955 and data["var56"] !=0)):
             s.append("2112")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]<-1.25382423 or data["var46"] ==0)):
             s.append("315")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]>=-1.25382423 and data["var46"] !=0) and (data["var52"]<-0.704087615 or data["var52"] ==0) and (data["var60"]<0.935133815 or data["var60"] ==0) and (data["var49"]<0.43300125 or data["var49"] ==0)):
             s.append("3101")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]>=-1.25382423 and data["var46"] !=0) and (data["var52"]<-0.704087615 or data["var52"] ==0) and (data["var60"]<0.935133815 or data["var60"] ==0) and (data["var49"]>=0.43300125 and data["var49"] !=0)):
             s.append("3102")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]>=-1.25382423 and data["var46"] !=0) and (data["var52"]<-0.704087615 or data["var52"] ==0) and (data["var60"]>=0.935133815 and data["var60"] !=0)):
             s.append("358")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]>=-1.25382423 and data["var46"] !=0) and (data["var52"]>=-0.704087615 and data["var52"] !=0) and (data["var24"]<0.053194046 or data["var24"] ==0) and (data["var15"]<-1.74312162 or data["var15"] ==0)):
             s.append("3103")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]>=-1.25382423 and data["var46"] !=0) and (data["var52"]>=-0.704087615 and data["var52"] !=0) and (data["var24"]<0.053194046 or data["var24"] ==0) and (data["var15"]>=-1.74312162 and data["var15"] !=0)):
             s.append("3104")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]>=-1.25382423 and data["var46"] !=0) and (data["var52"]>=-0.704087615 and data["var52"] !=0) and (data["var24"]>=0.053194046 and data["var24"] !=0) and (data["var53"]<1.3410157 or data["var53"] ==0)):
             s.append("3105")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]<0.335894614 or data["var29"] ==0) and (data["var46"]>=-1.25382423 and data["var46"] !=0) and (data["var52"]>=-0.704087615 and data["var52"] !=0) and (data["var24"]>=0.053194046 and data["var24"] !=0) and (data["var53"]>=1.3410157 and data["var53"] !=0)):
             s.append("3106")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]>=0.335894614 and data["var29"] !=0) and (data["var10"]<-1.85054207 or data["var10"] ==0)):
             s.append("317")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]>=0.335894614 and data["var29"] !=0) and (data["var10"]>=-1.85054207 and data["var10"] !=0) and (data["var42"]<2.17709303 or data["var42"] ==0)):
             s.append("333")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]<-2.92786741 or data["var19"] ==0) and (data["var29"]>=0.335894614 and data["var29"] !=0) and (data["var10"]>=-1.85054207 and data["var10"] !=0) and (data["var42"]>=2.17709303 and data["var42"] !=0)):
             s.append("334")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]<-1.85536492 or data["var05"] ==0) and (data["var48"]<2.45611286 or data["var48"] ==0) and (data["var26"]<-1.56381083 or data["var26"] ==0) and (data["var43"]<-0.340089262 or data["var43"] ==0)):
             s.append("361")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]<-1.85536492 or data["var05"] ==0) and (data["var48"]<2.45611286 or data["var48"] ==0) and (data["var26"]<-1.56381083 or data["var26"] ==0) and (data["var43"]>=-0.340089262 and data["var43"] !=0)):
             s.append("362")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]<-1.85536492 or data["var05"] ==0) and (data["var48"]<2.45611286 or data["var48"] ==0) and (data["var26"]>=-1.56381083 and data["var26"] !=0) and (data["var24"]<1.0933609 or data["var24"] ==0) and (data["var49"]<2.9429121 or data["var49"] ==0)):
             s.append("3107")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]<-1.85536492 or data["var05"] ==0) and (data["var48"]<2.45611286 or data["var48"] ==0) and (data["var26"]>=-1.56381083 and data["var26"] !=0) and (data["var24"]<1.0933609 or data["var24"] ==0) and (data["var49"]>=2.9429121 and data["var49"] !=0)):
             s.append("3108")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]<-1.85536492 or data["var05"] ==0) and (data["var48"]<2.45611286 or data["var48"] ==0) and (data["var26"]>=-1.56381083 and data["var26"] !=0) and (data["var24"]>=1.0933609 and data["var24"] !=0) and (data["var39"]<-0.0290128067 or data["var39"] ==0)):
             s.append("3109")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]<-1.85536492 or data["var05"] ==0) and (data["var48"]<2.45611286 or data["var48"] ==0) and (data["var26"]>=-1.56381083 and data["var26"] !=0) and (data["var24"]>=1.0933609 and data["var24"] !=0) and (data["var39"]>=-0.0290128067 and data["var39"] !=0)):
             s.append("3110")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]<-1.85536492 or data["var05"] ==0) and (data["var48"]>=2.45611286 and data["var48"] !=0)):
             s.append("320")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]<-1.28661919 or data["var02"] ==0) and (data["var25"]<-1.7003597 or data["var25"] ==0) and (data["var13"]<0.258240074 or data["var13"] ==0)):
             s.append("365")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]<-1.28661919 or data["var02"] ==0) and (data["var25"]<-1.7003597 or data["var25"] ==0) and (data["var13"]>=0.258240074 and data["var13"] !=0)):
             s.append("366")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]<-1.28661919 or data["var02"] ==0) and (data["var25"]>=-1.7003597 and data["var25"] !=0) and (data["var38"]<-0.0884047672 or data["var38"] ==0) and (data["var38"]<-1.11778998 or data["var38"] ==0)):
             s.append("3111")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]<-1.28661919 or data["var02"] ==0) and (data["var25"]>=-1.7003597 and data["var25"] !=0) and (data["var38"]<-0.0884047672 or data["var38"] ==0) and (data["var38"]>=-1.11778998 and data["var38"] !=0)):
             s.append("3112")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]<-1.28661919 or data["var02"] ==0) and (data["var25"]>=-1.7003597 and data["var25"] !=0) and (data["var38"]>=-0.0884047672 and data["var38"] !=0) and (data["var28"]<-1.14751148 or data["var28"] ==0)):
             s.append("3113")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]<-1.28661919 or data["var02"] ==0) and (data["var25"]>=-1.7003597 and data["var25"] !=0) and (data["var38"]>=-0.0884047672 and data["var38"] !=0) and (data["var28"]>=-1.14751148 and data["var28"] !=0)):
             s.append("3114")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]<0.923928142 or data["var05"] ==0) and (data["var57"]<-1.99139345 or data["var57"] ==0) and (data["var02"]<0.642190814 or data["var02"] ==0)):
             s.append("3115")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]<0.923928142 or data["var05"] ==0) and (data["var57"]<-1.99139345 or data["var57"] ==0) and (data["var02"]>=0.642190814 and data["var02"] !=0)):
             s.append("3116")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]<0.923928142 or data["var05"] ==0) and (data["var57"]>=-1.99139345 and data["var57"] !=0) and (data["var35"]<2.45979595 or data["var35"] ==0)):
             s.append("3117")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]<0.923928142 or data["var05"] ==0) and (data["var57"]>=-1.99139345 and data["var57"] !=0) and (data["var35"]>=2.45979595 and data["var35"] !=0)):
             s.append("3118")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]>=0.923928142 and data["var05"] !=0) and (data["var02"]<0.637635887 or data["var02"] ==0) and (data["var37"]<0.0834336057 or data["var37"] ==0)):
             s.append("3119")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]>=0.923928142 and data["var05"] !=0) and (data["var02"]<0.637635887 or data["var02"] ==0) and (data["var37"]>=0.0834336057 and data["var37"] !=0)):
             s.append("3120")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]>=0.923928142 and data["var05"] !=0) and (data["var02"]>=0.637635887 and data["var02"] !=0) and (data["var04"]<-1.16770315 or data["var04"] ==0)):
             s.append("3121")
    if((data["var19"]<-1.67156625 or data["var19"] ==0) and (data["var19"]>=-2.92786741 and data["var19"] !=0) and (data["var05"]>=-1.85536492 and data["var05"] !=0) and (data["var02"]>=-1.28661919 and data["var02"] !=0) and (data["var05"]>=0.923928142 and data["var05"] !=0) and (data["var02"]>=0.637635887 and data["var02"] !=0) and (data["var04"]>=-1.16770315 and data["var04"] !=0)):
             s.append("3122")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0) and (data["var01"]<-0.225437969 or data["var01"] ==0) and (data["var25"]<0.941757977 or data["var25"] ==0) and (data["var29"]<-0.73909831 or data["var29"] ==0)):
             s.append("3123")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0) and (data["var01"]<-0.225437969 or data["var01"] ==0) and (data["var25"]<0.941757977 or data["var25"] ==0) and (data["var29"]>=-0.73909831 and data["var29"] !=0)):
             s.append("3124")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0) and (data["var01"]<-0.225437969 or data["var01"] ==0) and (data["var25"]>=0.941757977 and data["var25"] !=0) and (data["var51"]<1.01445723 or data["var51"] ==0)):
             s.append("3125")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0) and (data["var01"]<-0.225437969 or data["var01"] ==0) and (data["var25"]>=0.941757977 and data["var25"] !=0) and (data["var51"]>=1.01445723 and data["var51"] !=0)):
             s.append("3126")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0) and (data["var01"]>=-0.225437969 and data["var01"] !=0) and (data["var36"]<-2.3103044 or data["var36"] ==0)):
             s.append("375")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0) and (data["var01"]>=-0.225437969 and data["var01"] !=0) and (data["var36"]>=-2.3103044 and data["var36"] !=0) and (data["var14"]<0.304157346 or data["var14"] ==0)):
             s.append("3127")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0) and (data["var01"]>=-0.225437969 and data["var01"] !=0) and (data["var36"]>=-2.3103044 and data["var36"] !=0) and (data["var14"]>=0.304157346 and data["var14"] !=0)):
             s.append("3128")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0) and (data["var42"]<-2.60553885 or data["var42"] ==0) and (data["var02"]<-1.77011538 or data["var02"] ==0)):
             s.append("377")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0) and (data["var42"]<-2.60553885 or data["var42"] ==0) and (data["var02"]>=-1.77011538 and data["var02"] !=0) and (data["var16"]<1.44890165 or data["var16"] ==0)):
             s.append("3129")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0) and (data["var42"]<-2.60553885 or data["var42"] ==0) and (data["var02"]>=-1.77011538 and data["var02"] !=0) and (data["var16"]>=1.44890165 and data["var16"] !=0)):
             s.append("3130")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0) and (data["var42"]>=-2.60553885 and data["var42"] !=0) and (data["var44"]<2.19004464 or data["var44"] ==0) and (data["var58"]<-2.08354855 or data["var58"] ==0)):
             s.append("3131")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0) and (data["var42"]>=-2.60553885 and data["var42"] !=0) and (data["var44"]<2.19004464 or data["var44"] ==0) and (data["var58"]>=-2.08354855 and data["var58"] !=0)):
             s.append("3132")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0) and (data["var42"]>=-2.60553885 and data["var42"] !=0) and (data["var44"]>=2.19004464 and data["var44"] !=0) and (data["var31"]<-1.19496226 or data["var31"] ==0)):
             s.append("3133")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]<1.79687166 or data["var55"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0) and (data["var42"]>=-2.60553885 and data["var42"] !=0) and (data["var44"]>=2.19004464 and data["var44"] !=0) and (data["var31"]>=-1.19496226 and data["var31"] !=0)):
             s.append("3134")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]<-1.59051871 or data["var21"] ==0) and (data["var08"]<-0.0547862798 or data["var08"] ==0) and (data["var57"]<0.885520697 or data["var57"] ==0) and (data["var40"]<0.54903692 or data["var40"] ==0)):
             s.append("3135")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]<-1.59051871 or data["var21"] ==0) and (data["var08"]<-0.0547862798 or data["var08"] ==0) and (data["var57"]<0.885520697 or data["var57"] ==0) and (data["var40"]>=0.54903692 and data["var40"] !=0)):
             s.append("3136")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]<-1.59051871 or data["var21"] ==0) and (data["var08"]<-0.0547862798 or data["var08"] ==0) and (data["var57"]>=0.885520697 and data["var57"] !=0) and (data["var41"]<-0.0609448776 or data["var41"] ==0)):
             s.append("3137")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]<-1.59051871 or data["var21"] ==0) and (data["var08"]<-0.0547862798 or data["var08"] ==0) and (data["var57"]>=0.885520697 and data["var57"] !=0) and (data["var41"]>=-0.0609448776 and data["var41"] !=0)):
             s.append("3138")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]<-1.59051871 or data["var21"] ==0) and (data["var08"]>=-0.0547862798 and data["var08"] !=0) and (data["var44"]<-1.32142115 or data["var44"] ==0)):
             s.append("383")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]<-1.59051871 or data["var21"] ==0) and (data["var08"]>=-0.0547862798 and data["var08"] !=0) and (data["var44"]>=-1.32142115 and data["var44"] !=0)):
             s.append("384")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]>=-1.59051871 and data["var21"] !=0) and (data["var55"]<3.54712677 or data["var55"] ==0) and (data["var23"]<-1.51653528 or data["var23"] ==0) and (data["var50"]<0.167935967 or data["var50"] ==0)):
             s.append("3139")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]>=-1.59051871 and data["var21"] !=0) and (data["var55"]<3.54712677 or data["var55"] ==0) and (data["var23"]<-1.51653528 or data["var23"] ==0) and (data["var50"]>=0.167935967 and data["var50"] !=0)):
             s.append("3140")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]>=-1.59051871 and data["var21"] !=0) and (data["var55"]<3.54712677 or data["var55"] ==0) and (data["var23"]>=-1.51653528 and data["var23"] !=0) and (data["var07"]<0.507454991 or data["var07"] ==0)):
             s.append("3141")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]>=-1.59051871 and data["var21"] !=0) and (data["var55"]<3.54712677 or data["var55"] ==0) and (data["var23"]>=-1.51653528 and data["var23"] !=0) and (data["var07"]>=0.507454991 and data["var07"] !=0)):
             s.append("3142")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]>=-1.59051871 and data["var21"] !=0) and (data["var55"]>=3.54712677 and data["var55"] !=0) and (data["var31"]<1.89968991 or data["var31"] ==0) and (data["var30"]<1.44825864 or data["var30"] ==0)):
             s.append("3143")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]>=-1.59051871 and data["var21"] !=0) and (data["var55"]>=3.54712677 and data["var55"] !=0) and (data["var31"]<1.89968991 or data["var31"] ==0) and (data["var30"]>=1.44825864 and data["var30"] !=0)):
             s.append("3144")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]<1.47024822 or data["var19"] ==0) and (data["var55"]>=1.79687166 and data["var55"] !=0) and (data["var21"]>=-1.59051871 and data["var21"] !=0) and (data["var55"]>=3.54712677 and data["var55"] !=0) and (data["var31"]>=1.89968991 and data["var31"] !=0)):
             s.append("388")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]<-1.6379627 or data["var24"] ==0) and (data["var30"]<0.713984132 or data["var30"] ==0) and (data["var12"]<0.624382794 or data["var12"] ==0)):
             s.append("389")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]<-1.6379627 or data["var24"] ==0) and (data["var30"]<0.713984132 or data["var30"] ==0) and (data["var12"]>=0.624382794 and data["var12"] !=0)):
             s.append("390")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]<-1.6379627 or data["var24"] ==0) and (data["var30"]>=0.713984132 and data["var30"] !=0)):
             s.append("350")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]>=-1.6379627 and data["var24"] !=0) and (data["var31"]<-0.743579865 or data["var31"] ==0) and (data["var17"]<-0.0894707814 or data["var17"] ==0)):
             s.append("391")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]>=-1.6379627 and data["var24"] !=0) and (data["var31"]<-0.743579865 or data["var31"] ==0) and (data["var17"]>=-0.0894707814 and data["var17"] !=0) and (data["var18"]<1.03565753 or data["var18"] ==0)):
             s.append("3145")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]>=-1.6379627 and data["var24"] !=0) and (data["var31"]<-0.743579865 or data["var31"] ==0) and (data["var17"]>=-0.0894707814 and data["var17"] !=0) and (data["var18"]>=1.03565753 and data["var18"] !=0)):
             s.append("3146")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]>=-1.6379627 and data["var24"] !=0) and (data["var31"]>=-0.743579865 and data["var31"] !=0) and (data["var60"]<-3.3635366 or data["var60"] ==0)):
             s.append("393")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]>=-1.6379627 and data["var24"] !=0) and (data["var31"]>=-0.743579865 and data["var31"] !=0) and (data["var60"]>=-3.3635366 and data["var60"] !=0) and (data["var05"]<2.2040782 or data["var05"] ==0)):
             s.append("3147")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]<-1.93377197 or data["var60"] ==0) and (data["var24"]>=-1.6379627 and data["var24"] !=0) and (data["var31"]>=-0.743579865 and data["var31"] !=0) and (data["var60"]>=-3.3635366 and data["var60"] !=0) and (data["var05"]>=2.2040782 and data["var05"] !=0)):
             s.append("3148")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]<-2.00310302 or data["var25"] ==0) and (data["var26"]<-2.20056629 or data["var26"] ==0)):
             s.append("353")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]<-2.00310302 or data["var25"] ==0) and (data["var26"]>=-2.20056629 and data["var26"] !=0) and (data["var03"]<-1.85088706 or data["var03"] ==0) and (data["var06"]<-0.0230635703 or data["var06"] ==0)):
             s.append("3149")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]<-2.00310302 or data["var25"] ==0) and (data["var26"]>=-2.20056629 and data["var26"] !=0) and (data["var03"]<-1.85088706 or data["var03"] ==0) and (data["var06"]>=-0.0230635703 and data["var06"] !=0)):
             s.append("3150")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]<-2.00310302 or data["var25"] ==0) and (data["var26"]>=-2.20056629 and data["var26"] !=0) and (data["var03"]>=-1.85088706 and data["var03"] !=0) and (data["var17"]<2.01787066 or data["var17"] ==0)):
             s.append("3151")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]<-2.00310302 or data["var25"] ==0) and (data["var26"]>=-2.20056629 and data["var26"] !=0) and (data["var03"]>=-1.85088706 and data["var03"] !=0) and (data["var17"]>=2.01787066 and data["var17"] !=0)):
             s.append("3152")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]<1.82016683 or data["var07"] ==0) and (data["var34"]<-2.73288107 or data["var34"] ==0) and (data["var29"]<-1.78592801 or data["var29"] ==0)):
             s.append("3153")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]<1.82016683 or data["var07"] ==0) and (data["var34"]<-2.73288107 or data["var34"] ==0) and (data["var29"]>=-1.78592801 and data["var29"] !=0)):
             s.append("3154")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]<1.82016683 or data["var07"] ==0) and (data["var34"]>=-2.73288107 and data["var34"] !=0) and (data["var28"]<0.468942732 or data["var28"] ==0)):
             s.append("3155")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]<1.82016683 or data["var07"] ==0) and (data["var34"]>=-2.73288107 and data["var34"] !=0) and (data["var28"]>=0.468942732 and data["var28"] !=0)):
             s.append("3156")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]>=1.82016683 and data["var07"] !=0) and (data["var25"]<-0.747435689 or data["var25"] ==0) and (data["var03"]<0.252803385 or data["var03"] ==0)):
             s.append("3157")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]>=1.82016683 and data["var07"] !=0) and (data["var25"]<-0.747435689 or data["var25"] ==0) and (data["var03"]>=0.252803385 and data["var03"] !=0)):
             s.append("3158")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]>=1.82016683 and data["var07"] !=0) and (data["var25"]>=-0.747435689 and data["var25"] !=0) and (data["var42"]<-2.76351523 or data["var42"] ==0)):
             s.append("3159")
    if((data["var19"]>=-1.67156625 and data["var19"] !=0) and (data["var19"]>=1.47024822 and data["var19"] !=0) and (data["var60"]>=-1.93377197 and data["var60"] !=0) and (data["var25"]>=-2.00310302 and data["var25"] !=0) and (data["var07"]>=1.82016683 and data["var07"] !=0) and (data["var25"]>=-0.747435689 and data["var25"] !=0) and (data["var42"]>=-2.76351523 and data["var42"] !=0)):
             s.append("3160")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]<-1.97760069 or data["var04"] ==0) and (data["var21"]<1.55994344 or data["var21"] ==0)):
             s.append("415")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]<-1.97760069 or data["var04"] ==0) and (data["var21"]>=1.55994344 and data["var21"] !=0)):
             s.append("416")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]<0.891286492 or data["var55"] ==0) and (data["var55"]<-1.30080628 or data["var55"] ==0) and (data["var18"]<0.976538301 or data["var18"] ==0)):
             s.append("485")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]<0.891286492 or data["var55"] ==0) and (data["var55"]<-1.30080628 or data["var55"] ==0) and (data["var18"]>=0.976538301 and data["var18"] !=0)):
             s.append("486")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]<0.891286492 or data["var55"] ==0) and (data["var55"]>=-1.30080628 and data["var55"] !=0) and (data["var06"]<-1.10522103 or data["var06"] ==0)):
             s.append("487")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]<0.891286492 or data["var55"] ==0) and (data["var55"]>=-1.30080628 and data["var55"] !=0) and (data["var06"]>=-1.10522103 and data["var06"] !=0)):
             s.append("488")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]>=0.891286492 and data["var55"] !=0) and (data["var33"]<-0.0886057913 or data["var33"] ==0) and (data["var02"]<-0.539194584 or data["var02"] ==0)):
             s.append("489")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]>=0.891286492 and data["var55"] !=0) and (data["var33"]<-0.0886057913 or data["var33"] ==0) and (data["var02"]>=-0.539194584 and data["var02"] !=0)):
             s.append("490")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]>=0.891286492 and data["var55"] !=0) and (data["var33"]>=-0.0886057913 and data["var33"] !=0) and (data["var07"]<2.48894 or data["var07"] ==0)):
             s.append("491")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]<1.23482442 or data["var04"] ==0) and (data["var55"]>=0.891286492 and data["var55"] !=0) and (data["var33"]>=-0.0886057913 and data["var33"] !=0) and (data["var07"]>=2.48894 and data["var07"] !=0)):
             s.append("492")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]>=1.23482442 and data["var04"] !=0) and (data["var24"]<-1.78399754 or data["var24"] ==0) and (data["var28"]<0.213465512 or data["var28"] ==0)):
             s.append("457")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]>=1.23482442 and data["var04"] !=0) and (data["var24"]<-1.78399754 or data["var24"] ==0) and (data["var28"]>=0.213465512 and data["var28"] !=0)):
             s.append("458")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]>=1.23482442 and data["var04"] !=0) and (data["var24"]>=-1.78399754 and data["var24"] !=0) and (data["var28"]<-1.1630969 or data["var28"] ==0) and (data["var13"]<-0.0682099387 or data["var13"] ==0)):
             s.append("493")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]>=1.23482442 and data["var04"] !=0) and (data["var24"]>=-1.78399754 and data["var24"] !=0) and (data["var28"]<-1.1630969 or data["var28"] ==0) and (data["var13"]>=-0.0682099387 and data["var13"] !=0)):
             s.append("494")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]>=1.23482442 and data["var04"] !=0) and (data["var24"]>=-1.78399754 and data["var24"] !=0) and (data["var28"]>=-1.1630969 and data["var28"] !=0) and (data["var33"]<-2.16205955 or data["var33"] ==0)):
             s.append("495")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var04"]>=-1.97760069 and data["var04"] !=0) and (data["var04"]>=1.23482442 and data["var04"] !=0) and (data["var24"]>=-1.78399754 and data["var24"] !=0) and (data["var28"]>=-1.1630969 and data["var28"] !=0) and (data["var33"]>=-2.16205955 and data["var33"] !=0)):
             s.append("496")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var11"]<-1.32635272 or data["var11"] ==0) and (data["var41"]<-0.267407268 or data["var41"] ==0)):
             s.append("419")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var11"]<-1.32635272 or data["var11"] ==0) and (data["var41"]>=-0.267407268 and data["var41"] !=0)):
             s.append("420")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var11"]>=-1.32635272 and data["var11"] !=0) and (data["var38"]<0.841527522 or data["var38"] ==0) and (data["var10"]<-1.48386192 or data["var10"] ==0)):
             s.append("435")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var11"]>=-1.32635272 and data["var11"] !=0) and (data["var38"]<0.841527522 or data["var38"] ==0) and (data["var10"]>=-1.48386192 and data["var10"] !=0)):
             s.append("436")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var11"]>=-1.32635272 and data["var11"] !=0) and (data["var38"]>=0.841527522 and data["var38"] !=0) and (data["var10"]<0.552159309 or data["var10"] ==0)):
             s.append("437")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var11"]>=-1.32635272 and data["var11"] !=0) and (data["var38"]>=0.841527522 and data["var38"] !=0) and (data["var10"]>=0.552159309 and data["var10"] !=0)):
             s.append("438")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-1.40789151 or data["var27"] ==0) and (data["var35"]<-1.40193963 or data["var35"] ==0) and (data["var10"]<-0.771188259 or data["var10"] ==0)):
             s.append("461")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-1.40789151 or data["var27"] ==0) and (data["var35"]<-1.40193963 or data["var35"] ==0) and (data["var10"]>=-0.771188259 and data["var10"] !=0)):
             s.append("462")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-1.40789151 or data["var27"] ==0) and (data["var35"]>=-1.40193963 and data["var35"] !=0) and (data["var05"]<-1.89779282 or data["var05"] ==0)):
             s.append("463")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-1.40789151 or data["var27"] ==0) and (data["var35"]>=-1.40193963 and data["var35"] !=0) and (data["var05"]>=-1.89779282 and data["var05"] !=0) and (data["var16"]<1.92670667 or data["var16"] ==0)):
             s.append("497")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]<-1.40789151 or data["var27"] ==0) and (data["var35"]>=-1.40193963 and data["var35"] !=0) and (data["var05"]>=-1.89779282 and data["var05"] !=0) and (data["var16"]>=1.92670667 and data["var16"] !=0)):
             s.append("498")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-1.40789151 and data["var27"] !=0) and (data["var04"]<-1.34350133 or data["var04"] ==0) and (data["var30"]<1.97323549 or data["var30"] ==0) and (data["var09"]<-2.00694346 or data["var09"] ==0)):
             s.append("499")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-1.40789151 and data["var27"] !=0) and (data["var04"]<-1.34350133 or data["var04"] ==0) and (data["var30"]<1.97323549 or data["var30"] ==0) and (data["var09"]>=-2.00694346 and data["var09"] !=0)):
             s.append("4100")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-1.40789151 and data["var27"] !=0) and (data["var04"]<-1.34350133 or data["var04"] ==0) and (data["var30"]>=1.97323549 and data["var30"] !=0)):
             s.append("466")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-1.40789151 and data["var27"] !=0) and (data["var04"]>=-1.34350133 and data["var04"] !=0) and (data["var26"]<2.09928226 or data["var26"] ==0) and (data["var45"]<1.67543364 or data["var45"] ==0)):
             s.append("4101")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-1.40789151 and data["var27"] !=0) and (data["var04"]>=-1.34350133 and data["var04"] !=0) and (data["var26"]<2.09928226 or data["var26"] ==0) and (data["var45"]>=1.67543364 and data["var45"] !=0)):
             s.append("4102")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]<-2.42058301 or data["var60"] ==0) and (data["var27"]>=-1.40789151 and data["var27"] !=0) and (data["var04"]>=-1.34350133 and data["var04"] !=0) and (data["var26"]>=2.09928226 and data["var26"] !=0)):
             s.append("468")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.61410141 or data["var26"] ==0) and (data["var30"]<0.0111082969 or data["var30"] ==0)):
             s.append("4103")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.61410141 or data["var26"] ==0) and (data["var30"]>=0.0111082969 and data["var30"] !=0)):
             s.append("4104")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.61410141 and data["var26"] !=0) and (data["var42"]<-1.2872262 or data["var42"] ==0)):
             s.append("4105")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.61410141 and data["var26"] !=0) and (data["var42"]>=-1.2872262 and data["var42"] !=0)):
             s.append("4106")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var58"]<-2.32160139 or data["var58"] ==0)):
             s.append("4107")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var58"]>=-2.32160139 and data["var58"] !=0)):
             s.append("4108")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var02"]<1.13211024 or data["var02"] ==0)):
             s.append("4109")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]<2.39354539 or data["var10"] ==0) and (data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var02"]>=1.13211024 and data["var02"] !=0)):
             s.append("4110")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.39354539 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]<-0.46432656 or data["var36"] ==0) and (data["var11"]<0.4417153 or data["var11"] ==0)):
             s.append("4111")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.39354539 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]<-0.46432656 or data["var36"] ==0) and (data["var11"]>=0.4417153 and data["var11"] !=0)):
             s.append("4112")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.39354539 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]>=-0.46432656 and data["var36"] !=0) and (data["var09"]<1.60609448 or data["var09"] ==0)):
             s.append("4113")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.39354539 and data["var10"] !=0) and (data["var10"]<3.56514144 or data["var10"] ==0) and (data["var36"]>=-0.46432656 and data["var36"] !=0) and (data["var09"]>=1.60609448 and data["var09"] !=0)):
             s.append("4114")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.39354539 and data["var10"] !=0) and (data["var10"]>=3.56514144 and data["var10"] !=0) and (data["var32"]<1.81330633 or data["var32"] ==0) and (data["var05"]<-1.60646081 or data["var05"] ==0)):
             s.append("4115")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.39354539 and data["var10"] !=0) and (data["var10"]>=3.56514144 and data["var10"] !=0) and (data["var32"]<1.81330633 or data["var32"] ==0) and (data["var05"]>=-1.60646081 and data["var05"] !=0)):
             s.append("4116")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]<2.50895834 or data["var47"] ==0) and (data["var60"]>=-2.42058301 and data["var60"] !=0) and (data["var10"]>=2.39354539 and data["var10"] !=0) and (data["var10"]>=3.56514144 and data["var10"] !=0) and (data["var32"]>=1.81330633 and data["var32"] !=0)):
             s.append("476")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]<0.668479681 or data["var01"] ==0) and (data["var19"]<-2.43883085 or data["var19"] ==0)):
             s.append("447")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]<0.668479681 or data["var01"] ==0) and (data["var19"]>=-2.43883085 and data["var19"] !=0) and (data["var56"]<-1.21146703 or data["var56"] ==0) and (data["var22"]<-0.316989809 or data["var22"] ==0)):
             s.append("4117")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]<0.668479681 or data["var01"] ==0) and (data["var19"]>=-2.43883085 and data["var19"] !=0) and (data["var56"]<-1.21146703 or data["var56"] ==0) and (data["var22"]>=-0.316989809 and data["var22"] !=0)):
             s.append("4118")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]<0.668479681 or data["var01"] ==0) and (data["var19"]>=-2.43883085 and data["var19"] !=0) and (data["var56"]>=-1.21146703 and data["var56"] !=0) and (data["var13"]<-2.08783937 or data["var13"] ==0)):
             s.append("4119")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]<0.668479681 or data["var01"] ==0) and (data["var19"]>=-2.43883085 and data["var19"] !=0) and (data["var56"]>=-1.21146703 and data["var56"] !=0) and (data["var13"]>=-2.08783937 and data["var13"] !=0)):
             s.append("4120")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]>=0.668479681 and data["var01"] !=0) and (data["var24"]<1.299613 or data["var24"] ==0)):
             s.append("449")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]>=0.668479681 and data["var01"] !=0) and (data["var24"]>=1.299613 and data["var24"] !=0) and (data["var47"]<2.96488905 or data["var47"] ==0)):
             s.append("479")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]<-0.143235564 or data["var43"] ==0) and (data["var01"]>=0.668479681 and data["var01"] !=0) and (data["var24"]>=1.299613 and data["var24"] !=0) and (data["var47"]>=2.96488905 and data["var47"] !=0)):
             s.append("480")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]<1.73241782 or data["var37"] ==0) and (data["var23"]<1.13893604 or data["var23"] ==0) and (data["var46"]<-1.14169812 or data["var46"] ==0) and (data["var46"]<-1.55322766 or data["var46"] ==0)):
             s.append("4121")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]<1.73241782 or data["var37"] ==0) and (data["var23"]<1.13893604 or data["var23"] ==0) and (data["var46"]<-1.14169812 or data["var46"] ==0) and (data["var46"]>=-1.55322766 and data["var46"] !=0)):
             s.append("4122")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]<1.73241782 or data["var37"] ==0) and (data["var23"]<1.13893604 or data["var23"] ==0) and (data["var46"]>=-1.14169812 and data["var46"] !=0) and (data["var54"]<1.55532265 or data["var54"] ==0)):
             s.append("4123")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]<1.73241782 or data["var37"] ==0) and (data["var23"]<1.13893604 or data["var23"] ==0) and (data["var46"]>=-1.14169812 and data["var46"] !=0) and (data["var54"]>=1.55532265 and data["var54"] !=0)):
             s.append("4124")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]<1.73241782 or data["var37"] ==0) and (data["var23"]>=1.13893604 and data["var23"] !=0) and (data["var43"]<1.76862454 or data["var43"] ==0) and (data["var49"]<-1.31276345 or data["var49"] ==0)):
             s.append("4125")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]<1.73241782 or data["var37"] ==0) and (data["var23"]>=1.13893604 and data["var23"] !=0) and (data["var43"]<1.76862454 or data["var43"] ==0) and (data["var49"]>=-1.31276345 and data["var49"] !=0)):
             s.append("4126")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]<1.73241782 or data["var37"] ==0) and (data["var23"]>=1.13893604 and data["var23"] !=0) and (data["var43"]>=1.76862454 and data["var43"] !=0)):
             s.append("484")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var47"]>=2.50895834 and data["var47"] !=0) and (data["var43"]>=-0.143235564 and data["var43"] !=0) and (data["var37"]>=1.73241782 and data["var37"] !=0)):
             s.append("430")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]<-1.11897635 or data["var54"] ==0) and (data["var13"]<-0.304727793 or data["var13"] ==0) and (data["var21"]<-1.62862325 or data["var21"] ==0)):
             s.append("545")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]<-1.11897635 or data["var54"] ==0) and (data["var13"]<-0.304727793 or data["var13"] ==0) and (data["var21"]>=-1.62862325 and data["var21"] !=0)):
             s.append("546")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]<-1.11897635 or data["var54"] ==0) and (data["var13"]>=-0.304727793 and data["var13"] !=0) and (data["var36"]<0.853780985 or data["var36"] ==0)):
             s.append("547")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]<-1.11897635 or data["var54"] ==0) and (data["var13"]>=-0.304727793 and data["var13"] !=0) and (data["var36"]>=0.853780985 and data["var36"] !=0)):
             s.append("548")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]>=-1.11897635 and data["var54"] !=0) and (data["var37"]<-0.0699489266 or data["var37"] ==0) and (data["var39"]<-1.34691656 or data["var39"] ==0) and (data["var15"]<-0.770819545 or data["var15"] ==0)):
             s.append("579")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]>=-1.11897635 and data["var54"] !=0) and (data["var37"]<-0.0699489266 or data["var37"] ==0) and (data["var39"]<-1.34691656 or data["var39"] ==0) and (data["var15"]>=-0.770819545 and data["var15"] !=0)):
             s.append("580")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]>=-1.11897635 and data["var54"] !=0) and (data["var37"]<-0.0699489266 or data["var37"] ==0) and (data["var39"]>=-1.34691656 and data["var39"] !=0) and (data["var03"]<0.658132553 or data["var03"] ==0)):
             s.append("581")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]>=-1.11897635 and data["var54"] !=0) and (data["var37"]<-0.0699489266 or data["var37"] ==0) and (data["var39"]>=-1.34691656 and data["var39"] !=0) and (data["var03"]>=0.658132553 and data["var03"] !=0)):
             s.append("582")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]>=-1.11897635 and data["var54"] !=0) and (data["var37"]>=-0.0699489266 and data["var37"] !=0) and (data["var10"]<1.50186157 or data["var10"] ==0) and (data["var56"]<-1.8453505 or data["var56"] ==0)):
             s.append("583")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]>=-1.11897635 and data["var54"] !=0) and (data["var37"]>=-0.0699489266 and data["var37"] !=0) and (data["var10"]<1.50186157 or data["var10"] ==0) and (data["var56"]>=-1.8453505 and data["var56"] !=0)):
             s.append("584")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]<-2.76766396 or data["var41"] ==0) and (data["var54"]>=-1.11897635 and data["var54"] !=0) and (data["var37"]>=-0.0699489266 and data["var37"] !=0) and (data["var10"]>=1.50186157 and data["var10"] !=0)):
             s.append("552")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]<-0.74508357 or data["var58"] ==0) and (data["var42"]<-0.0345157981 or data["var42"] ==0) and (data["var42"]<-0.920107603 or data["var42"] ==0)):
             s.append("585")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]<-0.74508357 or data["var58"] ==0) and (data["var42"]<-0.0345157981 or data["var42"] ==0) and (data["var42"]>=-0.920107603 and data["var42"] !=0)):
             s.append("586")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]<-0.74508357 or data["var58"] ==0) and (data["var42"]>=-0.0345157981 and data["var42"] !=0) and (data["var15"]<-0.974798679 or data["var15"] ==0)):
             s.append("587")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]<-0.74508357 or data["var58"] ==0) and (data["var42"]>=-0.0345157981 and data["var42"] !=0) and (data["var15"]>=-0.974798679 and data["var15"] !=0)):
             s.append("588")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]>=-0.74508357 and data["var58"] !=0) and (data["var39"]<-1.60226393 or data["var39"] ==0) and (data["var50"]<-2.03189087 or data["var50"] ==0)):
             s.append("589")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]>=-0.74508357 and data["var58"] !=0) and (data["var39"]<-1.60226393 or data["var39"] ==0) and (data["var50"]>=-2.03189087 and data["var50"] !=0)):
             s.append("590")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]>=-0.74508357 and data["var58"] !=0) and (data["var39"]>=-1.60226393 and data["var39"] !=0) and (data["var43"]<1.05688632 or data["var43"] ==0)):
             s.append("591")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]<-1.97528327 or data["var23"] ==0) and (data["var58"]>=-0.74508357 and data["var58"] !=0) and (data["var39"]>=-1.60226393 and data["var39"] !=0) and (data["var43"]>=1.05688632 and data["var43"] !=0)):
             s.append("592")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]<-1.84979582 or data["var17"] ==0) and (data["var28"]<-0.540824056 or data["var28"] ==0) and (data["var53"]<1.50492167 or data["var53"] ==0)):
             s.append("593")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]<-1.84979582 or data["var17"] ==0) and (data["var28"]<-0.540824056 or data["var28"] ==0) and (data["var53"]>=1.50492167 and data["var53"] !=0)):
             s.append("594")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]<-1.84979582 or data["var17"] ==0) and (data["var28"]>=-0.540824056 and data["var28"] !=0) and (data["var53"]<-1.04805624 or data["var53"] ==0)):
             s.append("595")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]<-1.84979582 or data["var17"] ==0) and (data["var28"]>=-0.540824056 and data["var28"] !=0) and (data["var53"]>=-1.04805624 and data["var53"] !=0)):
             s.append("596")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]>=-1.84979582 and data["var17"] !=0) and (data["var39"]<-2.42741585 or data["var39"] ==0) and (data["var47"]<-0.982910812 or data["var47"] ==0)):
             s.append("597")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]>=-1.84979582 and data["var17"] !=0) and (data["var39"]<-2.42741585 or data["var39"] ==0) and (data["var47"]>=-0.982910812 and data["var47"] !=0)):
             s.append("598")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]>=-1.84979582 and data["var17"] !=0) and (data["var39"]>=-2.42741585 and data["var39"] !=0) and (data["var40"]<-3.04525661 or data["var40"] ==0)):
             s.append("599")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]<1.46652865 or data["var23"] ==0) and (data["var41"]>=-2.76766396 and data["var41"] !=0) and (data["var23"]>=-1.97528327 and data["var23"] !=0) and (data["var17"]>=-1.84979582 and data["var17"] !=0) and (data["var39"]>=-2.42741585 and data["var39"] !=0) and (data["var40"]>=-3.04525661 and data["var40"] !=0)):
             s.append("5100")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]<-2.38698721 or data["var05"] ==0) and (data["var08"]<-1.39663148 or data["var08"] ==0)):
             s.append("535")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]<-2.38698721 or data["var05"] ==0) and (data["var08"]>=-1.39663148 and data["var08"] !=0) and (data["var53"]<-1.78270555 or data["var53"] ==0)):
             s.append("561")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]<-2.38698721 or data["var05"] ==0) and (data["var08"]>=-1.39663148 and data["var08"] !=0) and (data["var53"]>=-1.78270555 and data["var53"] !=0) and (data["var09"]<2.09654665 or data["var09"] ==0)):
             s.append("5101")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]<-2.38698721 or data["var05"] ==0) and (data["var08"]>=-1.39663148 and data["var08"] !=0) and (data["var53"]>=-1.78270555 and data["var53"] !=0) and (data["var09"]>=2.09654665 and data["var09"] !=0)):
             s.append("5102")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]<1.99113297 or data["var43"] ==0) and (data["var06"]<-2.45574117 or data["var06"] ==0) and (data["var02"]<-1.6523478 or data["var02"] ==0)):
             s.append("5103")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]<1.99113297 or data["var43"] ==0) and (data["var06"]<-2.45574117 or data["var06"] ==0) and (data["var02"]>=-1.6523478 and data["var02"] !=0)):
             s.append("5104")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]<1.99113297 or data["var43"] ==0) and (data["var06"]>=-2.45574117 and data["var06"] !=0) and (data["var09"]<-2.51162577 or data["var09"] ==0)):
             s.append("5105")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]<1.99113297 or data["var43"] ==0) and (data["var06"]>=-2.45574117 and data["var06"] !=0) and (data["var09"]>=-2.51162577 and data["var09"] !=0)):
             s.append("5106")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]>=1.99113297 and data["var43"] !=0) and (data["var53"]<-1.04695439 or data["var53"] ==0) and (data["var53"]<-1.30368829 or data["var53"] ==0)):
             s.append("5107")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]>=1.99113297 and data["var43"] !=0) and (data["var53"]<-1.04695439 or data["var53"] ==0) and (data["var53"]>=-1.30368829 and data["var53"] !=0)):
             s.append("5108")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]>=1.99113297 and data["var43"] !=0) and (data["var53"]>=-1.04695439 and data["var53"] !=0) and (data["var22"]<1.36044836 or data["var22"] ==0)):
             s.append("5109")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]<3.42494869 or data["var23"] ==0) and (data["var05"]>=-2.38698721 and data["var05"] !=0) and (data["var43"]>=1.99113297 and data["var43"] !=0) and (data["var53"]>=-1.04695439 and data["var53"] !=0) and (data["var22"]>=1.36044836 and data["var22"] !=0)):
             s.append("5110")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]>=3.42494869 and data["var23"] !=0) and (data["var17"]<2.187047 or data["var17"] ==0) and (data["var18"]<1.52286768 or data["var18"] ==0) and (data["var53"]<-1.83660936 or data["var53"] ==0)):
             s.append("567")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]>=3.42494869 and data["var23"] !=0) and (data["var17"]<2.187047 or data["var17"] ==0) and (data["var18"]<1.52286768 or data["var18"] ==0) and (data["var53"]>=-1.83660936 and data["var53"] !=0)):
             s.append("568")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]>=3.42494869 and data["var23"] !=0) and (data["var17"]<2.187047 or data["var17"] ==0) and (data["var18"]>=1.52286768 and data["var18"] !=0) and (data["var30"]<0.285538912 or data["var30"] ==0)):
             s.append("569")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]>=3.42494869 and data["var23"] !=0) and (data["var17"]<2.187047 or data["var17"] ==0) and (data["var18"]>=1.52286768 and data["var18"] !=0) and (data["var30"]>=0.285538912 and data["var30"] !=0)):
             s.append("570")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var23"]>=1.46652865 and data["var23"] !=0) and (data["var23"]>=3.42494869 and data["var23"] !=0) and (data["var17"]>=2.187047 and data["var17"] !=0)):
             s.append("522")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]<-1.06229091 or data["var49"] ==0) and (data["var40"]<-1.48795485 or data["var40"] ==0) and (data["var10"]<0.087829262 or data["var10"] ==0)):
             s.append("571")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]<-1.06229091 or data["var49"] ==0) and (data["var40"]<-1.48795485 or data["var40"] ==0) and (data["var10"]>=0.087829262 and data["var10"] !=0)):
             s.append("572")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]<-1.06229091 or data["var49"] ==0) and (data["var40"]>=-1.48795485 and data["var40"] !=0) and (data["var07"]<-1.69553804 or data["var07"] ==0)):
             s.append("573")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]<-1.06229091 or data["var49"] ==0) and (data["var40"]>=-1.48795485 and data["var40"] !=0) and (data["var07"]>=-1.69553804 and data["var07"] !=0) and (data["var12"]<-1.85146296 or data["var12"] ==0)):
             s.append("5111")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]<-1.06229091 or data["var49"] ==0) and (data["var40"]>=-1.48795485 and data["var40"] !=0) and (data["var07"]>=-1.69553804 and data["var07"] !=0) and (data["var12"]>=-1.85146296 and data["var12"] !=0)):
             s.append("5112")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]>=-1.06229091 and data["var49"] !=0) and (data["var24"]<1.34797657 or data["var24"] ==0) and (data["var51"]<-1.20025575 or data["var51"] ==0) and (data["var54"]<2.03248787 or data["var54"] ==0)):
             s.append("5113")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]>=-1.06229091 and data["var49"] !=0) and (data["var24"]<1.34797657 or data["var24"] ==0) and (data["var51"]<-1.20025575 or data["var51"] ==0) and (data["var54"]>=2.03248787 and data["var54"] !=0)):
             s.append("5114")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]>=-1.06229091 and data["var49"] !=0) and (data["var24"]<1.34797657 or data["var24"] ==0) and (data["var51"]>=-1.20025575 and data["var51"] !=0) and (data["var23"]<0.786075532 or data["var23"] ==0)):
             s.append("5115")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]>=-1.06229091 and data["var49"] !=0) and (data["var24"]<1.34797657 or data["var24"] ==0) and (data["var51"]>=-1.20025575 and data["var51"] !=0) and (data["var23"]>=0.786075532 and data["var23"] !=0)):
             s.append("5116")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]>=-1.06229091 and data["var49"] !=0) and (data["var24"]>=1.34797657 and data["var24"] !=0) and (data["var53"]<2.8605895 or data["var53"] ==0) and (data["var38"]<0.729619145 or data["var38"] ==0)):
             s.append("5117")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]>=-1.06229091 and data["var49"] !=0) and (data["var24"]>=1.34797657 and data["var24"] !=0) and (data["var53"]<2.8605895 or data["var53"] ==0) and (data["var38"]>=0.729619145 and data["var38"] !=0)):
             s.append("5118")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]<1.50965643 or data["var42"] ==0) and (data["var49"]>=-1.06229091 and data["var49"] !=0) and (data["var24"]>=1.34797657 and data["var24"] !=0) and (data["var53"]>=2.8605895 and data["var53"] !=0)):
             s.append("578")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]>=1.50965643 and data["var42"] !=0) and (data["var14"]<0.655697823 or data["var14"] ==0)):
             s.append("525")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]<1.80818224 or data["var42"] ==0) and (data["var42"]>=1.50965643 and data["var42"] !=0) and (data["var14"]>=0.655697823 and data["var14"] !=0)):
             s.append("526")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]>=1.80818224 and data["var42"] !=0) and (data["var02"]<1.63517237 or data["var02"] ==0)):
             s.append("513")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var42"]>=1.80818224 and data["var42"] !=0) and (data["var02"]>=1.63517237 and data["var02"] !=0)):
             s.append("514")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]<-1.93777347 or data["var04"] ==0) and (data["var58"]<0.774857163 or data["var58"] ==0)):
             s.append("629")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]<-1.93777347 or data["var04"] ==0) and (data["var58"]>=0.774857163 and data["var58"] !=0) and (data["var20"]<0.289601475 or data["var20"] ==0)):
             s.append("649")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]<-1.93777347 or data["var04"] ==0) and (data["var58"]>=0.774857163 and data["var58"] !=0) and (data["var20"]>=0.289601475 and data["var20"] !=0)):
             s.append("650")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]>=-1.93777347 and data["var04"] !=0) and (data["var13"]<1.96160376 or data["var13"] ==0) and (data["var44"]<1.99797094 or data["var44"] ==0) and (data["var48"]<1.29533887 or data["var48"] ==0)):
             s.append("679")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]>=-1.93777347 and data["var04"] !=0) and (data["var13"]<1.96160376 or data["var13"] ==0) and (data["var44"]<1.99797094 or data["var44"] ==0) and (data["var48"]>=1.29533887 and data["var48"] !=0)):
             s.append("680")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]>=-1.93777347 and data["var04"] !=0) and (data["var13"]<1.96160376 or data["var13"] ==0) and (data["var44"]>=1.99797094 and data["var44"] !=0) and (data["var17"]<0.50730145 or data["var17"] ==0)):
             s.append("681")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]>=-1.93777347 and data["var04"] !=0) and (data["var13"]<1.96160376 or data["var13"] ==0) and (data["var44"]>=1.99797094 and data["var44"] !=0) and (data["var17"]>=0.50730145 and data["var17"] !=0)):
             s.append("682")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]>=-1.93777347 and data["var04"] !=0) and (data["var13"]>=1.96160376 and data["var13"] !=0) and (data["var31"]<-1.02189183 or data["var31"] ==0)):
             s.append("653")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var04"]>=-1.93777347 and data["var04"] !=0) and (data["var13"]>=1.96160376 and data["var13"] !=0) and (data["var31"]>=-1.02189183 and data["var31"] !=0)):
             s.append("654")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]>=1.78389192 and data["var40"] !=0) and (data["var42"]<-1.23757434 or data["var42"] ==0)):
             s.append("617")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]<1.54023933 or data["var56"] ==0) and (data["var40"]>=1.78389192 and data["var40"] !=0) and (data["var42"]>=-1.23757434 and data["var42"] !=0)):
             s.append("618")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]>=1.54023933 and data["var56"] !=0) and (data["var17"]<-2.14339113 or data["var17"] ==0)):
             s.append("69")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]>=1.54023933 and data["var56"] !=0) and (data["var17"]>=-2.14339113 and data["var17"] !=0) and (data["var05"]<-1.81212807 or data["var05"] ==0)):
             s.append("619")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]>=1.54023933 and data["var56"] !=0) and (data["var17"]>=-2.14339113 and data["var17"] !=0) and (data["var05"]>=-1.81212807 and data["var05"] !=0) and (data["var14"]<2.45093608 or data["var14"] ==0) and (data["var02"]<-1.50196755 or data["var02"] ==0)):
             s.append("655")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]>=1.54023933 and data["var56"] !=0) and (data["var17"]>=-2.14339113 and data["var17"] !=0) and (data["var05"]>=-1.81212807 and data["var05"] !=0) and (data["var14"]<2.45093608 or data["var14"] ==0) and (data["var02"]>=-1.50196755 and data["var02"] !=0)):
             s.append("656")
    if((data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var56"]>=1.54023933 and data["var56"] !=0) and (data["var17"]>=-2.14339113 and data["var17"] !=0) and (data["var05"]>=-1.81212807 and data["var05"] !=0) and (data["var14"]>=2.45093608 and data["var14"] !=0)):
             s.append("634")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]<-1.89370942 or data["var01"] ==0) and (data["var27"]<1.41974497 or data["var27"] ==0) and (data["var17"]<-1.17959702 or data["var17"] ==0) and (data["var47"]<0.121644661 or data["var47"] ==0)):
             s.append("657")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]<-1.89370942 or data["var01"] ==0) and (data["var27"]<1.41974497 or data["var27"] ==0) and (data["var17"]<-1.17959702 or data["var17"] ==0) and (data["var47"]>=0.121644661 and data["var47"] !=0)):
             s.append("658")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]<-1.89370942 or data["var01"] ==0) and (data["var27"]<1.41974497 or data["var27"] ==0) and (data["var17"]>=-1.17959702 and data["var17"] !=0)):
             s.append("636")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]<-1.89370942 or data["var01"] ==0) and (data["var27"]>=1.41974497 and data["var27"] !=0)):
             s.append("622")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]<1.84471703 or data["var42"] ==0) and (data["var24"]<1.50616479 or data["var24"] ==0) and (data["var52"]<-1.98009515 or data["var52"] ==0)):
             s.append("683")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]<1.84471703 or data["var42"] ==0) and (data["var24"]<1.50616479 or data["var24"] ==0) and (data["var52"]>=-1.98009515 and data["var52"] !=0)):
             s.append("684")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]<1.84471703 or data["var42"] ==0) and (data["var24"]>=1.50616479 and data["var24"] !=0) and (data["var07"]<-0.775948822 or data["var07"] ==0)):
             s.append("685")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]<1.84471703 or data["var42"] ==0) and (data["var24"]>=1.50616479 and data["var24"] !=0) and (data["var07"]>=-0.775948822 and data["var07"] !=0)):
             s.append("686")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]>=1.84471703 and data["var42"] !=0) and (data["var32"]<-0.43428725 or data["var32"] ==0) and (data["var55"]<0.765713632 or data["var55"] ==0)):
             s.append("687")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]>=1.84471703 and data["var42"] !=0) and (data["var32"]<-0.43428725 or data["var32"] ==0) and (data["var55"]>=0.765713632 and data["var55"] !=0)):
             s.append("688")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]>=1.84471703 and data["var42"] !=0) and (data["var32"]>=-0.43428725 and data["var32"] !=0) and (data["var60"]<-1.11651778 or data["var60"] ==0)):
             s.append("689")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]<2.14551401 or data["var45"] ==0) and (data["var42"]>=1.84471703 and data["var42"] !=0) and (data["var32"]>=-0.43428725 and data["var32"] !=0) and (data["var60"]>=-1.11651778 and data["var60"] !=0)):
             s.append("690")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]>=2.14551401 and data["var45"] !=0) and (data["var17"]<1.21627319 or data["var17"] ==0)):
             s.append("639")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]<-1.98441625 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var45"]>=2.14551401 and data["var45"] !=0) and (data["var17"]>=1.21627319 and data["var17"] !=0)):
             s.append("640")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]<-1.81241477 or data["var05"] ==0) and (data["var55"]<1.85384369 or data["var55"] ==0) and (data["var03"]<1.60319805 or data["var03"] ==0) and (data["var43"]<1.8603313 or data["var43"] ==0)):
             s.append("691")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]<-1.81241477 or data["var05"] ==0) and (data["var55"]<1.85384369 or data["var55"] ==0) and (data["var03"]<1.60319805 or data["var03"] ==0) and (data["var43"]>=1.8603313 and data["var43"] !=0)):
             s.append("692")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]<-1.81241477 or data["var05"] ==0) and (data["var55"]<1.85384369 or data["var55"] ==0) and (data["var03"]>=1.60319805 and data["var03"] !=0) and (data["var16"]<-0.781519175 or data["var16"] ==0)):
             s.append("693")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]<-1.81241477 or data["var05"] ==0) and (data["var55"]<1.85384369 or data["var55"] ==0) and (data["var03"]>=1.60319805 and data["var03"] !=0) and (data["var16"]>=-0.781519175 and data["var16"] !=0)):
             s.append("694")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]<-1.81241477 or data["var05"] ==0) and (data["var55"]>=1.85384369 and data["var55"] !=0) and (data["var06"]<1.61670101 or data["var06"] ==0) and (data["var33"]<1.53028488 or data["var33"] ==0)):
             s.append("695")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]<-1.81241477 or data["var05"] ==0) and (data["var55"]>=1.85384369 and data["var55"] !=0) and (data["var06"]<1.61670101 or data["var06"] ==0) and (data["var33"]>=1.53028488 and data["var33"] !=0)):
             s.append("696")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]<-1.81241477 or data["var05"] ==0) and (data["var55"]>=1.85384369 and data["var55"] !=0) and (data["var06"]>=1.61670101 and data["var06"] !=0)):
             s.append("666")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]<1.84247065 or data["var10"] ==0) and (data["var10"]<-2.54790354 or data["var10"] ==0) and (data["var32"]<1.55237651 or data["var32"] ==0)):
             s.append("697")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]<1.84247065 or data["var10"] ==0) and (data["var10"]<-2.54790354 or data["var10"] ==0) and (data["var32"]>=1.55237651 and data["var32"] !=0)):
             s.append("698")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]<1.84247065 or data["var10"] ==0) and (data["var10"]>=-2.54790354 and data["var10"] !=0) and (data["var20"]<2.97110295 or data["var20"] ==0)):
             s.append("699")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]<1.84247065 or data["var10"] ==0) and (data["var10"]>=-2.54790354 and data["var10"] !=0) and (data["var20"]>=2.97110295 and data["var20"] !=0)):
             s.append("6100")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]>=1.84247065 and data["var10"] !=0) and (data["var38"]<-1.3646183 or data["var38"] ==0) and (data["var53"]<-2.32363725 or data["var53"] ==0)):
             s.append("6101")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]>=1.84247065 and data["var10"] !=0) and (data["var38"]<-1.3646183 or data["var38"] ==0) and (data["var53"]>=-2.32363725 and data["var53"] !=0)):
             s.append("6102")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]>=1.84247065 and data["var10"] !=0) and (data["var38"]>=-1.3646183 and data["var38"] !=0) and (data["var36"]<-0.521924436 or data["var36"] ==0)):
             s.append("6103")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]<2.23289204 or data["var33"] ==0) and (data["var05"]>=-1.81241477 and data["var05"] !=0) and (data["var10"]>=1.84247065 and data["var10"] !=0) and (data["var38"]>=-1.3646183 and data["var38"] !=0) and (data["var36"]>=-0.521924436 and data["var36"] !=0)):
             s.append("6104")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]<-1.0294137 or data["var30"] ==0) and (data["var44"]<1.34060001 or data["var44"] ==0) and (data["var04"]<-1.46490693 or data["var04"] ==0)):
             s.append("671")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]<-1.0294137 or data["var30"] ==0) and (data["var44"]<1.34060001 or data["var44"] ==0) and (data["var04"]>=-1.46490693 and data["var04"] !=0) and (data["var17"]<2.21926165 or data["var17"] ==0)):
             s.append("6105")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]<-1.0294137 or data["var30"] ==0) and (data["var44"]<1.34060001 or data["var44"] ==0) and (data["var04"]>=-1.46490693 and data["var04"] !=0) and (data["var17"]>=2.21926165 and data["var17"] !=0)):
             s.append("6106")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]<-1.0294137 or data["var30"] ==0) and (data["var44"]>=1.34060001 and data["var44"] !=0) and (data["var24"]<0.373994946 or data["var24"] ==0)):
             s.append("673")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]<-1.0294137 or data["var30"] ==0) and (data["var44"]>=1.34060001 and data["var44"] !=0) and (data["var24"]>=0.373994946 and data["var24"] !=0)):
             s.append("674")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]<0.748870432 or data["var18"] ==0) and (data["var14"]<-1.274665 or data["var14"] ==0) and (data["var44"]<0.253341258 or data["var44"] ==0)):
             s.append("6107")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]<0.748870432 or data["var18"] ==0) and (data["var14"]<-1.274665 or data["var14"] ==0) and (data["var44"]>=0.253341258 and data["var44"] !=0)):
             s.append("6108")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]<0.748870432 or data["var18"] ==0) and (data["var14"]>=-1.274665 and data["var14"] !=0) and (data["var34"]<1.17698348 or data["var34"] ==0)):
             s.append("6109")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]<0.748870432 or data["var18"] ==0) and (data["var14"]>=-1.274665 and data["var14"] !=0) and (data["var34"]>=1.17698348 and data["var34"] !=0)):
             s.append("6110")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]>=0.748870432 and data["var18"] !=0) and (data["var51"]<-0.582453072 or data["var51"] ==0) and (data["var28"]<2.25535488 or data["var28"] ==0)):
             s.append("6111")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]>=0.748870432 and data["var18"] !=0) and (data["var51"]<-0.582453072 or data["var51"] ==0) and (data["var28"]>=2.25535488 and data["var28"] !=0)):
             s.append("6112")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]>=0.748870432 and data["var18"] !=0) and (data["var51"]>=-0.582453072 and data["var51"] !=0) and (data["var55"]<0.473362982 or data["var55"] ==0)):
             s.append("6113")
    if((data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var30"]>=-1.98441625 and data["var30"] !=0) and (data["var33"]>=2.23289204 and data["var33"] !=0) and (data["var30"]>=-1.0294137 and data["var30"] !=0) and (data["var18"]>=0.748870432 and data["var18"] !=0) and (data["var51"]>=-0.582453072 and data["var51"] !=0) and (data["var55"]>=0.473362982 and data["var55"] !=0)):
             s.append("6114")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]<-1.34910941 or data["var27"] ==0) and (data["var04"]<-2.2324357 or data["var04"] ==0)):
             s.append("77")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]<-1.34910941 or data["var27"] ==0) and (data["var04"]>=-2.2324357 and data["var04"] !=0) and (data["var38"]<-3.68100691 or data["var38"] ==0)):
             s.append("715")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]<-1.34910941 or data["var27"] ==0) and (data["var04"]>=-2.2324357 and data["var04"] !=0) and (data["var38"]>=-3.68100691 and data["var38"] !=0)):
             s.append("716")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]<-1.22746348 or data["var31"] ==0) and (data["var07"]<1.09611559 or data["var07"] ==0) and (data["var34"]<1.34896374 or data["var34"] ==0) and (data["var37"]<-1.45182991 or data["var37"] ==0)):
             s.append("747")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]<-1.22746348 or data["var31"] ==0) and (data["var07"]<1.09611559 or data["var07"] ==0) and (data["var34"]<1.34896374 or data["var34"] ==0) and (data["var37"]>=-1.45182991 and data["var37"] !=0)):
             s.append("748")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]<-1.22746348 or data["var31"] ==0) and (data["var07"]<1.09611559 or data["var07"] ==0) and (data["var34"]>=1.34896374 and data["var34"] !=0) and (data["var17"]<-0.147800446 or data["var17"] ==0)):
             s.append("749")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]<-1.22746348 or data["var31"] ==0) and (data["var07"]<1.09611559 or data["var07"] ==0) and (data["var34"]>=1.34896374 and data["var34"] !=0) and (data["var17"]>=-0.147800446 and data["var17"] !=0)):
             s.append("750")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]<-1.22746348 or data["var31"] ==0) and (data["var07"]>=1.09611559 and data["var07"] !=0)):
             s.append("718")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]<-0.97549516 or data["var21"] ==0) and (data["var52"]<1.00194848 or data["var52"] ==0) and (data["var28"]<1.97203159 or data["var28"] ==0) and (data["var08"]<-1.27258873 or data["var08"] ==0)):
             s.append("781")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]<-0.97549516 or data["var21"] ==0) and (data["var52"]<1.00194848 or data["var52"] ==0) and (data["var28"]<1.97203159 or data["var28"] ==0) and (data["var08"]>=-1.27258873 and data["var08"] !=0)):
             s.append("782")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]<-0.97549516 or data["var21"] ==0) and (data["var52"]<1.00194848 or data["var52"] ==0) and (data["var28"]>=1.97203159 and data["var28"] !=0)):
             s.append("752")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]<-0.97549516 or data["var21"] ==0) and (data["var52"]>=1.00194848 and data["var52"] !=0) and (data["var59"]<-0.438346505 or data["var59"] ==0)):
             s.append("753")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]<-0.97549516 or data["var21"] ==0) and (data["var52"]>=1.00194848 and data["var52"] !=0) and (data["var59"]>=-0.438346505 and data["var59"] !=0) and (data["var38"]<-2.85011649 or data["var38"] ==0)):
             s.append("783")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]<-0.97549516 or data["var21"] ==0) and (data["var52"]>=1.00194848 and data["var52"] !=0) and (data["var59"]>=-0.438346505 and data["var59"] !=0) and (data["var38"]>=-2.85011649 and data["var38"] !=0)):
             s.append("784")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]>=-0.97549516 and data["var21"] !=0) and (data["var14"]<-1.52697265 or data["var14"] ==0) and (data["var20"]<1.03829646 or data["var20"] ==0)):
             s.append("755")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]>=-0.97549516 and data["var21"] !=0) and (data["var14"]<-1.52697265 or data["var14"] ==0) and (data["var20"]>=1.03829646 and data["var20"] !=0)):
             s.append("756")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]>=-0.97549516 and data["var21"] !=0) and (data["var14"]>=-1.52697265 and data["var14"] !=0) and (data["var17"]<-1.41012108 or data["var17"] ==0) and (data["var06"]<0.423474133 or data["var06"] ==0)):
             s.append("785")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]>=-0.97549516 and data["var21"] !=0) and (data["var14"]>=-1.52697265 and data["var14"] !=0) and (data["var17"]<-1.41012108 or data["var17"] ==0) and (data["var06"]>=0.423474133 and data["var06"] !=0)):
             s.append("786")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]>=-0.97549516 and data["var21"] !=0) and (data["var14"]>=-1.52697265 and data["var14"] !=0) and (data["var17"]>=-1.41012108 and data["var17"] !=0) and (data["var33"]<1.54285192 or data["var33"] ==0)):
             s.append("787")
    if((data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var27"]>=-1.34910941 and data["var27"] !=0) and (data["var31"]>=-1.22746348 and data["var31"] !=0) and (data["var21"]>=-0.97549516 and data["var21"] !=0) and (data["var14"]>=-1.52697265 and data["var14"] !=0) and (data["var17"]>=-1.41012108 and data["var17"] !=0) and (data["var33"]>=1.54285192 and data["var33"] !=0)):
             s.append("788")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]<-0.771112323 or data["var44"] ==0) and (data["var49"]<-0.404628247 or data["var49"] ==0) and (data["var21"]<-1.84456658 or data["var21"] ==0)):
             s.append("759")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]<-0.771112323 or data["var44"] ==0) and (data["var49"]<-0.404628247 or data["var49"] ==0) and (data["var21"]>=-1.84456658 and data["var21"] !=0) and (data["var27"]<2.33782721 or data["var27"] ==0)):
             s.append("789")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]<-0.771112323 or data["var44"] ==0) and (data["var49"]<-0.404628247 or data["var49"] ==0) and (data["var21"]>=-1.84456658 and data["var21"] !=0) and (data["var27"]>=2.33782721 and data["var27"] !=0)):
             s.append("790")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]<-0.771112323 or data["var44"] ==0) and (data["var49"]>=-0.404628247 and data["var49"] !=0) and (data["var37"]<0.448862851 or data["var37"] ==0) and (data["var06"]<-0.00766754849 or data["var06"] ==0)):
             s.append("791")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]<-0.771112323 or data["var44"] ==0) and (data["var49"]>=-0.404628247 and data["var49"] !=0) and (data["var37"]<0.448862851 or data["var37"] ==0) and (data["var06"]>=-0.00766754849 and data["var06"] !=0)):
             s.append("792")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]<-0.771112323 or data["var44"] ==0) and (data["var49"]>=-0.404628247 and data["var49"] !=0) and (data["var37"]>=0.448862851 and data["var37"] !=0) and (data["var15"]<-1.43631554 or data["var15"] ==0)):
             s.append("793")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]<-0.771112323 or data["var44"] ==0) and (data["var49"]>=-0.404628247 and data["var49"] !=0) and (data["var37"]>=0.448862851 and data["var37"] !=0) and (data["var15"]>=-1.43631554 and data["var15"] !=0)):
             s.append("794")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]<-1.89425611 or data["var04"] ==0) and (data["var60"]<-0.567750096 or data["var60"] ==0) and (data["var09"]<0.158299267 or data["var09"] ==0)):
             s.append("795")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]<-1.89425611 or data["var04"] ==0) and (data["var60"]<-0.567750096 or data["var60"] ==0) and (data["var09"]>=0.158299267 and data["var09"] !=0)):
             s.append("796")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]<-1.89425611 or data["var04"] ==0) and (data["var60"]>=-0.567750096 and data["var60"] !=0) and (data["var29"]<-3.510777 or data["var29"] ==0)):
             s.append("797")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]<-1.89425611 or data["var04"] ==0) and (data["var60"]>=-0.567750096 and data["var60"] !=0) and (data["var29"]>=-3.510777 and data["var29"] !=0)):
             s.append("798")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]>=-1.89425611 and data["var04"] !=0) and (data["var04"]<0.882614136 or data["var04"] ==0) and (data["var55"]<2.22414017 or data["var55"] ==0)):
             s.append("799")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]>=-1.89425611 and data["var04"] !=0) and (data["var04"]<0.882614136 or data["var04"] ==0) and (data["var55"]>=2.22414017 and data["var55"] !=0)):
             s.append("7100")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]>=-1.89425611 and data["var04"] !=0) and (data["var04"]>=0.882614136 and data["var04"] !=0) and (data["var27"]<-1.45499444 or data["var27"] ==0)):
             s.append("7101")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]<-2.21378732 or data["var29"] ==0) and (data["var44"]>=-0.771112323 and data["var44"] !=0) and (data["var04"]>=-1.89425611 and data["var04"] !=0) and (data["var04"]>=0.882614136 and data["var04"] !=0) and (data["var27"]>=-1.45499444 and data["var27"] !=0)):
             s.append("7102")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]<1.67582774 or data["var55"] ==0) and (data["var02"]<2.08554554 or data["var02"] ==0) and (data["var39"]<1.31975484 or data["var39"] ==0)):
             s.append("7103")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]<1.67582774 or data["var55"] ==0) and (data["var02"]<2.08554554 or data["var02"] ==0) and (data["var39"]>=1.31975484 and data["var39"] !=0)):
             s.append("7104")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]<1.67582774 or data["var55"] ==0) and (data["var02"]>=2.08554554 and data["var02"] !=0) and (data["var26"]<0.243830532 or data["var26"] ==0)):
             s.append("7105")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]<1.67582774 or data["var55"] ==0) and (data["var02"]>=2.08554554 and data["var02"] !=0) and (data["var26"]>=0.243830532 and data["var26"] !=0)):
             s.append("7106")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]>=1.67582774 and data["var55"] !=0) and (data["var34"]<-0.16853644 or data["var34"] ==0) and (data["var47"]<-1.95889068 or data["var47"] ==0)):
             s.append("7107")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]>=1.67582774 and data["var55"] !=0) and (data["var34"]<-0.16853644 or data["var34"] ==0) and (data["var47"]>=-1.95889068 and data["var47"] !=0)):
             s.append("7108")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]>=1.67582774 and data["var55"] !=0) and (data["var34"]>=-0.16853644 and data["var34"] !=0) and (data["var15"]<0.90023011 or data["var15"] ==0)):
             s.append("7109")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]<-1.62503552 or data["var06"] ==0) and (data["var55"]>=1.67582774 and data["var55"] !=0) and (data["var34"]>=-0.16853644 and data["var34"] !=0) and (data["var15"]>=0.90023011 and data["var15"] !=0)):
             s.append("7110")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]<-2.62729383 or data["var41"] ==0) and (data["var24"]<0.549886942 or data["var24"] ==0) and (data["var15"]<-1.34436631 or data["var15"] ==0)):
             s.append("7111")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]<-2.62729383 or data["var41"] ==0) and (data["var24"]<0.549886942 or data["var24"] ==0) and (data["var15"]>=-1.34436631 and data["var15"] !=0)):
             s.append("7112")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]<-2.62729383 or data["var41"] ==0) and (data["var24"]>=0.549886942 and data["var24"] !=0) and (data["var37"]<-0.471444368 or data["var37"] ==0)):
             s.append("7113")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]<-2.62729383 or data["var41"] ==0) and (data["var24"]>=0.549886942 and data["var24"] !=0) and (data["var37"]>=-0.471444368 and data["var37"] !=0)):
             s.append("7114")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]>=-2.62729383 and data["var41"] !=0) and (data["var19"]<-2.46206594 or data["var19"] ==0) and (data["var41"]<-2.29104543 or data["var41"] ==0)):
             s.append("7115")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]>=-2.62729383 and data["var41"] !=0) and (data["var19"]<-2.46206594 or data["var19"] ==0) and (data["var41"]>=-2.29104543 and data["var41"] !=0)):
             s.append("7116")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]>=-2.62729383 and data["var41"] !=0) and (data["var19"]>=-2.46206594 and data["var19"] !=0) and (data["var20"]<-2.01797724 or data["var20"] ==0)):
             s.append("7117")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]<2.01861525 or data["var29"] ==0) and (data["var29"]>=-2.21378732 and data["var29"] !=0) and (data["var06"]>=-1.62503552 and data["var06"] !=0) and (data["var41"]>=-2.62729383 and data["var41"] !=0) and (data["var19"]>=-2.46206594 and data["var19"] !=0) and (data["var20"]>=-2.01797724 and data["var20"] !=0)):
             s.append("7118")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]<-1.83648634 or data["var60"] ==0) and (data["var55"]<2.24617457 or data["var55"] ==0) and (data["var40"]<-0.629433393 or data["var40"] ==0) and (data["var31"]<1.39934707 or data["var31"] ==0) and (data["var42"]<-0.57039392 or data["var42"] ==0)):
             s.append("7119")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]<-1.83648634 or data["var60"] ==0) and (data["var55"]<2.24617457 or data["var55"] ==0) and (data["var40"]<-0.629433393 or data["var40"] ==0) and (data["var31"]<1.39934707 or data["var31"] ==0) and (data["var42"]>=-0.57039392 and data["var42"] !=0)):
             s.append("7120")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]<-1.83648634 or data["var60"] ==0) and (data["var55"]<2.24617457 or data["var55"] ==0) and (data["var40"]<-0.629433393 or data["var40"] ==0) and (data["var31"]>=1.39934707 and data["var31"] !=0)):
             s.append("776")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]<-1.83648634 or data["var60"] ==0) and (data["var55"]<2.24617457 or data["var55"] ==0) and (data["var40"]>=-0.629433393 and data["var40"] !=0)):
             s.append("744")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]<-1.83648634 or data["var60"] ==0) and (data["var55"]>=2.24617457 and data["var55"] !=0)):
             s.append("726")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]>=-1.83648634 and data["var60"] !=0) and (data["var39"]<2.72590494 or data["var39"] ==0) and (data["var58"]<2.44823265 or data["var58"] ==0) and (data["var32"]<1.75630236 or data["var32"] ==0) and (data["var55"]<1.69511199 or data["var55"] ==0)):
             s.append("7121")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]>=-1.83648634 and data["var60"] !=0) and (data["var39"]<2.72590494 or data["var39"] ==0) and (data["var58"]<2.44823265 or data["var58"] ==0) and (data["var32"]<1.75630236 or data["var32"] ==0) and (data["var55"]>=1.69511199 and data["var55"] !=0)):
             s.append("7122")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]>=-1.83648634 and data["var60"] !=0) and (data["var39"]<2.72590494 or data["var39"] ==0) and (data["var58"]<2.44823265 or data["var58"] ==0) and (data["var32"]>=1.75630236 and data["var32"] !=0) and (data["var20"]<0.527863085 or data["var20"] ==0)):
             s.append("7123")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]>=-1.83648634 and data["var60"] !=0) and (data["var39"]<2.72590494 or data["var39"] ==0) and (data["var58"]<2.44823265 or data["var58"] ==0) and (data["var32"]>=1.75630236 and data["var32"] !=0) and (data["var20"]>=0.527863085 and data["var20"] !=0)):
             s.append("7124")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]>=-1.83648634 and data["var60"] !=0) and (data["var39"]<2.72590494 or data["var39"] ==0) and (data["var58"]>=2.44823265 and data["var58"] !=0) and (data["var01"]<-1.60909939 or data["var01"] ==0)):
             s.append("779")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]>=-1.83648634 and data["var60"] !=0) and (data["var39"]<2.72590494 or data["var39"] ==0) and (data["var58"]>=2.44823265 and data["var58"] !=0) and (data["var01"]>=-1.60909939 and data["var01"] !=0)):
             s.append("780")
    if((data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var29"]>=2.01861525 and data["var29"] !=0) and (data["var60"]>=-1.83648634 and data["var60"] !=0) and (data["var39"]>=2.72590494 and data["var39"] !=0)):
             s.append("728")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]<-1.24643636 or data["var35"] ==0) and (data["var45"]<2.064888 or data["var45"] ==0) and (data["var54"]<2.25802708 or data["var54"] ==0) and (data["var03"]<1.99905849 or data["var03"] ==0)):
             s.append("827")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]<-1.24643636 or data["var35"] ==0) and (data["var45"]<2.064888 or data["var45"] ==0) and (data["var54"]<2.25802708 or data["var54"] ==0) and (data["var03"]>=1.99905849 and data["var03"] !=0)):
             s.append("828")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]<-1.24643636 or data["var35"] ==0) and (data["var45"]<2.064888 or data["var45"] ==0) and (data["var54"]>=2.25802708 and data["var54"] !=0)):
             s.append("816")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]<-1.24643636 or data["var35"] ==0) and (data["var45"]>=2.064888 and data["var45"] !=0)):
             s.append("88")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]<-1.30538988 or data["var17"] ==0) and (data["var59"]<1.56881452 or data["var59"] ==0)):
             s.append("817")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]<-1.30538988 or data["var17"] ==0) and (data["var59"]>=1.56881452 and data["var59"] !=0)):
             s.append("818")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]<1.84026766 or data["var38"] ==0) and (data["var35"]<-0.0358979553 or data["var35"] ==0) and (data["var04"]<1.13308692 or data["var04"] ==0) and (data["var50"]<0.954692483 or data["var50"] ==0)):
             s.append("873")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]<1.84026766 or data["var38"] ==0) and (data["var35"]<-0.0358979553 or data["var35"] ==0) and (data["var04"]<1.13308692 or data["var04"] ==0) and (data["var50"]>=0.954692483 and data["var50"] !=0)):
             s.append("874")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]<1.84026766 or data["var38"] ==0) and (data["var35"]<-0.0358979553 or data["var35"] ==0) and (data["var04"]>=1.13308692 and data["var04"] !=0)):
             s.append("846")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]<1.84026766 or data["var38"] ==0) and (data["var35"]>=-0.0358979553 and data["var35"] !=0) and (data["var17"]<-0.15849103 or data["var17"] ==0) and (data["var20"]<-0.933584273 or data["var20"] ==0)):
             s.append("875")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]<1.84026766 or data["var38"] ==0) and (data["var35"]>=-0.0358979553 and data["var35"] !=0) and (data["var17"]<-0.15849103 or data["var17"] ==0) and (data["var20"]>=-0.933584273 and data["var20"] !=0)):
             s.append("876")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]<1.84026766 or data["var38"] ==0) and (data["var35"]>=-0.0358979553 and data["var35"] !=0) and (data["var17"]>=-0.15849103 and data["var17"] !=0) and (data["var40"]<-0.0135378931 or data["var40"] ==0)):
             s.append("877")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]<1.84026766 or data["var38"] ==0) and (data["var35"]>=-0.0358979553 and data["var35"] !=0) and (data["var17"]>=-0.15849103 and data["var17"] !=0) and (data["var40"]>=-0.0135378931 and data["var40"] !=0)):
             s.append("878")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]>=1.84026766 and data["var38"] !=0) and (data["var52"]<-0.137582645 or data["var52"] ==0)):
             s.append("831")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var35"]>=-1.24643636 and data["var35"] !=0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var38"]>=1.84026766 and data["var38"] !=0) and (data["var52"]>=-0.137582645 and data["var52"] !=0)):
             s.append("832")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]<-1.4013927 or data["var39"] ==0) and (data["var11"]<-2.79243469 or data["var11"] ==0) and (data["var05"]<0.928247094 or data["var05"] ==0)):
             s.append("849")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]<-1.4013927 or data["var39"] ==0) and (data["var11"]<-2.79243469 or data["var11"] ==0) and (data["var05"]>=0.928247094 and data["var05"] !=0)):
             s.append("850")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]<-1.4013927 or data["var39"] ==0) and (data["var11"]>=-2.79243469 and data["var11"] !=0) and (data["var23"]<1.6145072 or data["var23"] ==0) and (data["var50"]<-1.22239566 or data["var50"] ==0)):
             s.append("879")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]<-1.4013927 or data["var39"] ==0) and (data["var11"]>=-2.79243469 and data["var11"] !=0) and (data["var23"]<1.6145072 or data["var23"] ==0) and (data["var50"]>=-1.22239566 and data["var50"] !=0)):
             s.append("880")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]<-1.4013927 or data["var39"] ==0) and (data["var11"]>=-2.79243469 and data["var11"] !=0) and (data["var23"]>=1.6145072 and data["var23"] !=0)):
             s.append("852")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]<1.14194095 or data["var44"] ==0) and (data["var38"]<1.19993758 or data["var38"] ==0) and (data["var20"]<-0.738651037 or data["var20"] ==0)):
             s.append("881")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]<1.14194095 or data["var44"] ==0) and (data["var38"]<1.19993758 or data["var38"] ==0) and (data["var20"]>=-0.738651037 and data["var20"] !=0)):
             s.append("882")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]<1.14194095 or data["var44"] ==0) and (data["var38"]>=1.19993758 and data["var38"] !=0) and (data["var20"]<0.9377141 or data["var20"] ==0)):
             s.append("883")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]<1.14194095 or data["var44"] ==0) and (data["var38"]>=1.19993758 and data["var38"] !=0) and (data["var20"]>=0.9377141 and data["var20"] !=0)):
             s.append("884")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]>=1.14194095 and data["var44"] !=0) and (data["var39"]<-1.18078637 or data["var39"] ==0) and (data["var37"]<-0.0493448712 or data["var37"] ==0)):
             s.append("885")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]>=1.14194095 and data["var44"] !=0) and (data["var39"]<-1.18078637 or data["var39"] ==0) and (data["var37"]>=-0.0493448712 and data["var37"] !=0)):
             s.append("886")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]>=1.14194095 and data["var44"] !=0) and (data["var39"]>=-1.18078637 and data["var39"] !=0) and (data["var26"]<-0.0647158846 or data["var26"] ==0)):
             s.append("887")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]<2.70191455 or data["var25"] ==0) and (data["var39"]>=-1.4013927 and data["var39"] !=0) and (data["var44"]>=1.14194095 and data["var44"] !=0) and (data["var39"]>=-1.18078637 and data["var39"] !=0) and (data["var26"]>=-0.0647158846 and data["var26"] !=0)):
             s.append("888")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]<-1.63107133 or data["var11"] ==0) and (data["var25"]>=2.70191455 and data["var25"] !=0)):
             s.append("812")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]<-2.78669119 or data["var43"] ==0) and (data["var21"]<1.58640218 or data["var21"] ==0) and (data["var49"]<-2.45162749 or data["var49"] ==0)):
             s.append("889")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]<-2.78669119 or data["var43"] ==0) and (data["var21"]<1.58640218 or data["var21"] ==0) and (data["var49"]>=-2.45162749 and data["var49"] !=0)):
             s.append("890")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]<-2.78669119 or data["var43"] ==0) and (data["var21"]>=1.58640218 and data["var21"] !=0) and (data["var10"]<0.602792561 or data["var10"] ==0)):
             s.append("891")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]<-2.78669119 or data["var43"] ==0) and (data["var21"]>=1.58640218 and data["var21"] !=0) and (data["var10"]>=0.602792561 and data["var10"] !=0)):
             s.append("892")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]>=-2.78669119 and data["var43"] !=0) and (data["var31"]<3.05502629 or data["var31"] ==0) and (data["var40"]<-1.80883455 or data["var40"] ==0)):
             s.append("893")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]>=-2.78669119 and data["var43"] !=0) and (data["var31"]<3.05502629 or data["var31"] ==0) and (data["var40"]>=-1.80883455 and data["var40"] !=0)):
             s.append("894")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]>=-2.78669119 and data["var43"] !=0) and (data["var31"]>=3.05502629 and data["var31"] !=0) and (data["var33"]<1.39353144 or data["var33"] ==0)):
             s.append("895")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]<1.49942529 or data["var59"] ==0) and (data["var43"]>=-2.78669119 and data["var43"] !=0) and (data["var31"]>=3.05502629 and data["var31"] !=0) and (data["var33"]>=1.39353144 and data["var33"] !=0)):
             s.append("896")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]<-1.57058144 or data["var02"] ==0) and (data["var11"]<-1.0622716 or data["var11"] ==0) and (data["var57"]<-0.0892354697 or data["var57"] ==0)):
             s.append("897")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]<-1.57058144 or data["var02"] ==0) and (data["var11"]<-1.0622716 or data["var11"] ==0) and (data["var57"]>=-0.0892354697 and data["var57"] !=0)):
             s.append("898")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]<-1.57058144 or data["var02"] ==0) and (data["var11"]>=-1.0622716 and data["var11"] !=0) and (data["var21"]<1.31448388 or data["var21"] ==0)):
             s.append("899")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]<-1.57058144 or data["var02"] ==0) and (data["var11"]>=-1.0622716 and data["var11"] !=0) and (data["var21"]>=1.31448388 and data["var21"] !=0)):
             s.append("8100")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]>=-1.57058144 and data["var02"] !=0) and (data["var22"]<-1.73384953 or data["var22"] ==0) and (data["var38"]<-0.96816051 or data["var38"] ==0)):
             s.append("8101")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]>=-1.57058144 and data["var02"] !=0) and (data["var22"]<-1.73384953 or data["var22"] ==0) and (data["var38"]>=-0.96816051 and data["var38"] !=0)):
             s.append("8102")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]>=-1.57058144 and data["var02"] !=0) and (data["var22"]>=-1.73384953 and data["var22"] !=0) and (data["var25"]<1.23263454 or data["var25"] ==0)):
             s.append("8103")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]<1.86143088 or data["var11"] ==0) and (data["var59"]>=1.49942529 and data["var59"] !=0) and (data["var02"]>=-1.57058144 and data["var02"] !=0) and (data["var22"]>=-1.73384953 and data["var22"] !=0) and (data["var25"]>=1.23263454 and data["var25"] !=0)):
             s.append("8104")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]<-1.17809379 or data["var41"] ==0) and (data["var54"]<0.354200512 or data["var54"] ==0) and (data["var28"]<2.23073578 or data["var28"] ==0)):
             s.append("8105")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]<-1.17809379 or data["var41"] ==0) and (data["var54"]<0.354200512 or data["var54"] ==0) and (data["var28"]>=2.23073578 and data["var28"] !=0)):
             s.append("8106")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]<-1.17809379 or data["var41"] ==0) and (data["var54"]>=0.354200512 and data["var54"] !=0) and (data["var54"]<0.527061164 or data["var54"] ==0)):
             s.append("8107")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]<-1.17809379 or data["var41"] ==0) and (data["var54"]>=0.354200512 and data["var54"] !=0) and (data["var54"]>=0.527061164 and data["var54"] !=0)):
             s.append("8108")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]>=-1.17809379 and data["var41"] !=0) and (data["var04"]<-1.83763504 or data["var04"] ==0) and (data["var42"]<1.5444665 or data["var42"] ==0)):
             s.append("8109")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]>=-1.17809379 and data["var41"] !=0) and (data["var04"]<-1.83763504 or data["var04"] ==0) and (data["var42"]>=1.5444665 and data["var42"] !=0)):
             s.append("8110")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]>=-1.17809379 and data["var41"] !=0) and (data["var04"]>=-1.83763504 and data["var04"] !=0) and (data["var15"]<-0.974770069 or data["var15"] ==0)):
             s.append("8111")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]<3.48958921 or data["var11"] ==0) and (data["var41"]>=-1.17809379 and data["var41"] !=0) and (data["var04"]>=-1.83763504 and data["var04"] !=0) and (data["var15"]>=-0.974770069 and data["var15"] !=0)):
             s.append("8112")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]>=3.48958921 and data["var11"] !=0) and (data["var26"]<1.47544158 or data["var26"] ==0) and (data["var07"]<1.10488939 or data["var07"] ==0)):
             s.append("869")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]>=3.48958921 and data["var11"] !=0) and (data["var26"]<1.47544158 or data["var26"] ==0) and (data["var07"]>=1.10488939 and data["var07"] !=0)):
             s.append("870")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]>=3.48958921 and data["var11"] !=0) and (data["var26"]>=1.47544158 and data["var26"] !=0) and (data["var31"]<0.0410354286 or data["var31"] ==0)):
             s.append("871")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var11"]>=-1.63107133 and data["var11"] !=0) and (data["var11"]>=1.86143088 and data["var11"] !=0) and (data["var11"]>=3.48958921 and data["var11"] !=0) and (data["var26"]>=1.47544158 and data["var26"] !=0) and (data["var31"]>=0.0410354286 and data["var31"] !=0)):
             s.append("872")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]<-1.51738191 or data["var57"] ==0) and (data["var48"]<-1.81506658 or data["var48"] ==0) and (data["var47"]<0.301285654 or data["var47"] ==0)):
             s.append("997")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]<-1.51738191 or data["var57"] ==0) and (data["var48"]<-1.81506658 or data["var48"] ==0) and (data["var47"]>=0.301285654 and data["var47"] !=0)):
             s.append("998")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]<-1.51738191 or data["var57"] ==0) and (data["var48"]>=-1.81506658 and data["var48"] !=0) and (data["var30"]<2.25202847 or data["var30"] ==0)):
             s.append("999")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]<-1.51738191 or data["var57"] ==0) and (data["var48"]>=-1.81506658 and data["var48"] !=0) and (data["var30"]>=2.25202847 and data["var30"] !=0)):
             s.append("9100")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]>=-1.51738191 and data["var57"] !=0) and (data["var31"]<-1.26611507 or data["var31"] ==0) and (data["var44"]<2.27965617 or data["var44"] ==0)):
             s.append("9101")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]>=-1.51738191 and data["var57"] !=0) and (data["var31"]<-1.26611507 or data["var31"] ==0) and (data["var44"]>=2.27965617 and data["var44"] !=0)):
             s.append("9102")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]>=-1.51738191 and data["var57"] !=0) and (data["var31"]>=-1.26611507 and data["var31"] !=0) and (data["var55"]<-2.25984001 or data["var55"] ==0)):
             s.append("9103")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]<1.79143107 or data["var04"] ==0) and (data["var57"]>=-1.51738191 and data["var57"] !=0) and (data["var31"]>=-1.26611507 and data["var31"] !=0) and (data["var55"]>=-2.25984001 and data["var55"] !=0)):
             s.append("9104")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]<2.97559071 or data["var04"] ==0) and (data["var37"]<-2.02459908 or data["var37"] ==0) and (data["var38"]<1.37907982 or data["var38"] ==0)):
             s.append("9105")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]<2.97559071 or data["var04"] ==0) and (data["var37"]<-2.02459908 or data["var37"] ==0) and (data["var38"]>=1.37907982 and data["var38"] !=0)):
             s.append("9106")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]<2.97559071 or data["var04"] ==0) and (data["var37"]>=-2.02459908 and data["var37"] !=0) and (data["var01"]<-1.53167224 or data["var01"] ==0)):
             s.append("9107")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]<2.97559071 or data["var04"] ==0) and (data["var37"]>=-2.02459908 and data["var37"] !=0) and (data["var01"]>=-1.53167224 and data["var01"] !=0)):
             s.append("9108")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]>=2.97559071 and data["var04"] !=0) and (data["var08"]<-0.708910167 or data["var08"] ==0) and (data["var35"]<1.07038283 or data["var35"] ==0)):
             s.append("9109")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]>=2.97559071 and data["var04"] !=0) and (data["var08"]<-0.708910167 or data["var08"] ==0) and (data["var35"]>=1.07038283 and data["var35"] !=0)):
             s.append("9110")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]>=2.97559071 and data["var04"] !=0) and (data["var08"]>=-0.708910167 and data["var08"] !=0) and (data["var08"]<0.137109846 or data["var08"] ==0)):
             s.append("9111")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]<2.40679455 or data["var53"] ==0) and (data["var04"]>=1.79143107 and data["var04"] !=0) and (data["var04"]>=2.97559071 and data["var04"] !=0) and (data["var08"]>=-0.708910167 and data["var08"] !=0) and (data["var08"]>=0.137109846 and data["var08"] !=0)):
             s.append("9112")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]<-0.720267773 or data["var09"] ==0) and (data["var44"]<-0.233131871 or data["var44"] ==0) and (data["var51"]<0.39728868 or data["var51"] ==0)):
             s.append("9113")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]<-0.720267773 or data["var09"] ==0) and (data["var44"]<-0.233131871 or data["var44"] ==0) and (data["var51"]>=0.39728868 and data["var51"] !=0)):
             s.append("9114")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]<-0.720267773 or data["var09"] ==0) and (data["var44"]>=-0.233131871 and data["var44"] !=0) and (data["var20"]<0.794072986 or data["var20"] ==0)):
             s.append("9115")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]<-0.720267773 or data["var09"] ==0) and (data["var44"]>=-0.233131871 and data["var44"] !=0) and (data["var20"]>=0.794072986 and data["var20"] !=0)):
             s.append("9116")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]>=-0.720267773 and data["var09"] !=0) and (data["var48"]<1.35724568 or data["var48"] ==0) and (data["var12"]<1.54731083 or data["var12"] ==0)):
             s.append("9117")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]>=-0.720267773 and data["var09"] !=0) and (data["var48"]<1.35724568 or data["var48"] ==0) and (data["var12"]>=1.54731083 and data["var12"] !=0)):
             s.append("9118")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]>=-0.720267773 and data["var09"] !=0) and (data["var48"]>=1.35724568 and data["var48"] !=0) and (data["var03"]<1.37324262 or data["var03"] ==0)):
             s.append("9119")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]<-0.626977682 or data["var16"] ==0) and (data["var09"]>=-0.720267773 and data["var09"] !=0) and (data["var48"]>=1.35724568 and data["var48"] !=0) and (data["var03"]>=1.37324262 and data["var03"] !=0)):
             s.append("9120")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]>=-0.626977682 and data["var16"] !=0) and (data["var50"]<1.71608341 or data["var50"] ==0) and (data["var16"]<0.821819544 or data["var16"] ==0) and (data["var31"]<1.05041683 or data["var31"] ==0)):
             s.append("9121")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]>=-0.626977682 and data["var16"] !=0) and (data["var50"]<1.71608341 or data["var50"] ==0) and (data["var16"]<0.821819544 or data["var16"] ==0) and (data["var31"]>=1.05041683 and data["var31"] !=0)):
             s.append("9122")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]>=-0.626977682 and data["var16"] !=0) and (data["var50"]<1.71608341 or data["var50"] ==0) and (data["var16"]>=0.821819544 and data["var16"] !=0) and (data["var59"]<-1.51924157 or data["var59"] ==0)):
             s.append("9123")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]>=-0.626977682 and data["var16"] !=0) and (data["var50"]<1.71608341 or data["var50"] ==0) and (data["var16"]>=0.821819544 and data["var16"] !=0) and (data["var59"]>=-1.51924157 and data["var59"] !=0)):
             s.append("9124")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]>=-0.626977682 and data["var16"] !=0) and (data["var50"]>=1.71608341 and data["var50"] !=0) and (data["var28"]<1.17316914 or data["var28"] ==0)):
             s.append("969")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]<2.20871353 or data["var13"] ==0) and (data["var53"]>=2.40679455 and data["var53"] !=0) and (data["var16"]>=-0.626977682 and data["var16"] !=0) and (data["var50"]>=1.71608341 and data["var50"] !=0) and (data["var28"]>=1.17316914 and data["var28"] !=0)):
             s.append("970")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]<-1.04591393 or data["var08"] ==0) and (data["var45"]<-0.156746313 or data["var45"] ==0) and (data["var56"]<0.746018291 or data["var56"] ==0)):
             s.append("971")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]<-1.04591393 or data["var08"] ==0) and (data["var45"]<-0.156746313 or data["var45"] ==0) and (data["var56"]>=0.746018291 and data["var56"] !=0)):
             s.append("972")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]<-1.04591393 or data["var08"] ==0) and (data["var45"]>=-0.156746313 and data["var45"] !=0) and (data["var50"]<0.643118262 or data["var50"] ==0) and (data["var39"]<-0.292360902 or data["var39"] ==0)):
             s.append("9125")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]<-1.04591393 or data["var08"] ==0) and (data["var45"]>=-0.156746313 and data["var45"] !=0) and (data["var50"]<0.643118262 or data["var50"] ==0) and (data["var39"]>=-0.292360902 and data["var39"] !=0)):
             s.append("9126")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]<-1.04591393 or data["var08"] ==0) and (data["var45"]>=-0.156746313 and data["var45"] !=0) and (data["var50"]>=0.643118262 and data["var50"] !=0)):
             s.append("974")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]>=-1.04591393 and data["var08"] !=0) and (data["var57"]<1.13372862 or data["var57"] ==0) and (data["var47"]<0.8178128 or data["var47"] ==0) and (data["var55"]<-2.03654718 or data["var55"] ==0)):
             s.append("9127")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]>=-1.04591393 and data["var08"] !=0) and (data["var57"]<1.13372862 or data["var57"] ==0) and (data["var47"]<0.8178128 or data["var47"] ==0) and (data["var55"]>=-2.03654718 and data["var55"] !=0)):
             s.append("9128")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]>=-1.04591393 and data["var08"] !=0) and (data["var57"]<1.13372862 or data["var57"] ==0) and (data["var47"]>=0.8178128 and data["var47"] !=0) and (data["var60"]<0.635138094 or data["var60"] ==0)):
             s.append("9129")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]>=-1.04591393 and data["var08"] !=0) and (data["var57"]<1.13372862 or data["var57"] ==0) and (data["var47"]>=0.8178128 and data["var47"] !=0) and (data["var60"]>=0.635138094 and data["var60"] !=0)):
             s.append("9130")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]>=-1.04591393 and data["var08"] !=0) and (data["var57"]>=1.13372862 and data["var57"] !=0) and (data["var56"]<-0.875224948 or data["var56"] ==0)):
             s.append("977")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]<-0.899486303 or data["var06"] ==0) and (data["var08"]>=-1.04591393 and data["var08"] !=0) and (data["var57"]>=1.13372862 and data["var57"] !=0) and (data["var56"]>=-0.875224948 and data["var56"] !=0)):
             s.append("978")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]<-1.11997199 or data["var56"] ==0) and (data["var47"]<-1.03704011 or data["var47"] ==0) and (data["var06"]<0.387880385 or data["var06"] ==0)):
             s.append("979")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]<-1.11997199 or data["var56"] ==0) and (data["var47"]<-1.03704011 or data["var47"] ==0) and (data["var06"]>=0.387880385 and data["var06"] !=0) and (data["var04"]<-0.490572512 or data["var04"] ==0)):
             s.append("9131")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]<-1.11997199 or data["var56"] ==0) and (data["var47"]<-1.03704011 or data["var47"] ==0) and (data["var06"]>=0.387880385 and data["var06"] !=0) and (data["var04"]>=-0.490572512 and data["var04"] !=0)):
             s.append("9132")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]<-1.11997199 or data["var56"] ==0) and (data["var47"]>=-1.03704011 and data["var47"] !=0) and (data["var13"]<3.64394689 or data["var13"] ==0) and (data["var10"]<-1.49343574 or data["var10"] ==0)):
             s.append("9133")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]<-1.11997199 or data["var56"] ==0) and (data["var47"]>=-1.03704011 and data["var47"] !=0) and (data["var13"]<3.64394689 or data["var13"] ==0) and (data["var10"]>=-1.49343574 and data["var10"] !=0)):
             s.append("9134")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]<-1.11997199 or data["var56"] ==0) and (data["var47"]>=-1.03704011 and data["var47"] !=0) and (data["var13"]>=3.64394689 and data["var13"] !=0)):
             s.append("982")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]<1.00981045 or data["var55"] ==0) and (data["var36"]<1.71804166 or data["var36"] ==0) and (data["var47"]<1.79132867 or data["var47"] ==0)):
             s.append("9135")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]<1.00981045 or data["var55"] ==0) and (data["var36"]<1.71804166 or data["var36"] ==0) and (data["var47"]>=1.79132867 and data["var47"] !=0)):
             s.append("9136")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]<1.00981045 or data["var55"] ==0) and (data["var36"]>=1.71804166 and data["var36"] !=0) and (data["var60"]<-0.0843080506 or data["var60"] ==0)):
             s.append("9137")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]<1.00981045 or data["var55"] ==0) and (data["var36"]>=1.71804166 and data["var36"] !=0) and (data["var60"]>=-0.0843080506 and data["var60"] !=0)):
             s.append("9138")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]>=1.00981045 and data["var55"] !=0) and (data["var51"]<-0.361987889 or data["var51"] ==0) and (data["var49"]<-0.877824903 or data["var49"] ==0)):
             s.append("9139")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]>=1.00981045 and data["var55"] !=0) and (data["var51"]<-0.361987889 or data["var51"] ==0) and (data["var49"]>=-0.877824903 and data["var49"] !=0)):
             s.append("9140")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]>=1.00981045 and data["var55"] !=0) and (data["var51"]>=-0.361987889 and data["var51"] !=0) and (data["var17"]<0.512655079 or data["var17"] ==0)):
             s.append("9141")
    if((data["var57"]<1.65416396 or data["var57"] ==0) and (data["var13"]>=2.20871353 and data["var13"] !=0) and (data["var06"]>=-0.899486303 and data["var06"] !=0) and (data["var56"]>=-1.11997199 and data["var56"] !=0) and (data["var55"]>=1.00981045 and data["var55"] !=0) and (data["var51"]>=-0.361987889 and data["var51"] !=0) and (data["var17"]>=0.512655079 and data["var17"] !=0)):
             s.append("9142")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]<-2.04285717 or data["var07"] ==0) and (data["var50"]<0.764625251 or data["var50"] ==0) and (data["var19"]<1.61559498 or data["var19"] ==0)):
             s.append("923")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]<-2.04285717 or data["var07"] ==0) and (data["var50"]<0.764625251 or data["var50"] ==0) and (data["var19"]>=1.61559498 and data["var19"] !=0)):
             s.append("924")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]<-2.04285717 or data["var07"] ==0) and (data["var50"]>=0.764625251 and data["var50"] !=0) and (data["var11"]<-0.24093464 or data["var11"] ==0)):
             s.append("925")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]<-2.04285717 or data["var07"] ==0) and (data["var50"]>=0.764625251 and data["var50"] !=0) and (data["var11"]>=-0.24093464 and data["var11"] !=0) and (data["var17"]<-1.21179044 or data["var17"] ==0)):
             s.append("947")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]<-2.04285717 or data["var07"] ==0) and (data["var50"]>=0.764625251 and data["var50"] !=0) and (data["var11"]>=-0.24093464 and data["var11"] !=0) and (data["var17"]>=-1.21179044 and data["var17"] !=0)):
             s.append("948")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]<-2.1972754 or data["var47"] ==0) and (data["var53"]<-1.8506906 or data["var53"] ==0)):
             s.append("927")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]<-2.1972754 or data["var47"] ==0) and (data["var53"]>=-1.8506906 and data["var53"] !=0) and (data["var01"]<-1.79601979 or data["var01"] ==0) and (data["var33"]<0.459835619 or data["var33"] ==0)):
             s.append("987")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]<-2.1972754 or data["var47"] ==0) and (data["var53"]>=-1.8506906 and data["var53"] !=0) and (data["var01"]<-1.79601979 or data["var01"] ==0) and (data["var33"]>=0.459835619 and data["var33"] !=0)):
             s.append("988")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]<-2.1972754 or data["var47"] ==0) and (data["var53"]>=-1.8506906 and data["var53"] !=0) and (data["var01"]>=-1.79601979 and data["var01"] !=0) and (data["var27"]<-2.12782478 or data["var27"] ==0)):
             s.append("989")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]<-2.1972754 or data["var47"] ==0) and (data["var53"]>=-1.8506906 and data["var53"] !=0) and (data["var01"]>=-1.79601979 and data["var01"] !=0) and (data["var27"]>=-2.12782478 and data["var27"] !=0) and (data["var44"]<-1.20554972 or data["var44"] ==0)):
             s.append("9143")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]<-2.1972754 or data["var47"] ==0) and (data["var53"]>=-1.8506906 and data["var53"] !=0) and (data["var01"]>=-1.79601979 and data["var01"] !=0) and (data["var27"]>=-2.12782478 and data["var27"] !=0) and (data["var44"]>=-1.20554972 and data["var44"] !=0)):
             s.append("9144")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]<-1.99483871 or data["var28"] ==0) and (data["var55"]<2.10744095 or data["var55"] ==0) and (data["var29"]<-1.54710197 or data["var29"] ==0) and (data["var29"]<-2.16131973 or data["var29"] ==0)):
             s.append("9145")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]<-1.99483871 or data["var28"] ==0) and (data["var55"]<2.10744095 or data["var55"] ==0) and (data["var29"]<-1.54710197 or data["var29"] ==0) and (data["var29"]>=-2.16131973 and data["var29"] !=0)):
             s.append("9146")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]<-1.99483871 or data["var28"] ==0) and (data["var55"]<2.10744095 or data["var55"] ==0) and (data["var29"]>=-1.54710197 and data["var29"] !=0) and (data["var45"]<-1.39954567 or data["var45"] ==0)):
             s.append("9147")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]<-1.99483871 or data["var28"] ==0) and (data["var55"]<2.10744095 or data["var55"] ==0) and (data["var29"]>=-1.54710197 and data["var29"] !=0) and (data["var45"]>=-1.39954567 and data["var45"] !=0)):
             s.append("9148")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]<-1.99483871 or data["var28"] ==0) and (data["var55"]>=2.10744095 and data["var55"] !=0)):
             s.append("952")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]<-2.39980888 or data["var38"] ==0) and (data["var52"]<0.489794612 or data["var52"] ==0) and (data["var55"]<-1.6724596 or data["var55"] ==0)):
             s.append("9149")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]<-2.39980888 or data["var38"] ==0) and (data["var52"]<0.489794612 or data["var52"] ==0) and (data["var55"]>=-1.6724596 and data["var55"] !=0)):
             s.append("9150")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]<-2.39980888 or data["var38"] ==0) and (data["var52"]>=0.489794612 and data["var52"] !=0) and (data["var35"]<0.0212926194 or data["var35"] ==0)):
             s.append("9151")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]<-2.39980888 or data["var38"] ==0) and (data["var52"]>=0.489794612 and data["var52"] !=0) and (data["var35"]>=0.0212926194 and data["var35"] !=0)):
             s.append("9152")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]>=-2.39980888 and data["var38"] !=0) and (data["var39"]<-1.75669813 or data["var39"] ==0) and (data["var29"]<-0.128107756 or data["var29"] ==0)):
             s.append("9153")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]>=-2.39980888 and data["var38"] !=0) and (data["var39"]<-1.75669813 or data["var39"] ==0) and (data["var29"]>=-0.128107756 and data["var29"] !=0)):
             s.append("9154")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]>=-2.39980888 and data["var38"] !=0) and (data["var39"]>=-1.75669813 and data["var39"] !=0) and (data["var57"]<2.92998815 or data["var57"] ==0)):
             s.append("9155")
    if((data["var57"]>=1.65416396 and data["var57"] !=0) and (data["var07"]>=-2.04285717 and data["var07"] !=0) and (data["var47"]>=-2.1972754 and data["var47"] !=0) and (data["var28"]>=-1.99483871 and data["var28"] !=0) and (data["var38"]>=-2.39980888 and data["var38"] !=0) and (data["var39"]>=-1.75669813 and data["var39"] !=0) and (data["var57"]>=2.92998815 and data["var57"] !=0)):
             s.append("9156")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]<-2.02413225 or data["var10"] ==0) and (data["var49"]<0.623034239 or data["var49"] ==0)):
             s.append("1031")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]<-2.02413225 or data["var10"] ==0) and (data["var49"]>=0.623034239 and data["var49"] !=0) and (data["var53"]<-0.341107845 or data["var53"] ==0)):
             s.append("1055")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]<-2.02413225 or data["var10"] ==0) and (data["var49"]>=0.623034239 and data["var49"] !=0) and (data["var53"]>=-0.341107845 and data["var53"] !=0)):
             s.append("1056")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]>=-2.02413225 and data["var10"] !=0) and (data["var07"]<-2.46717882 or data["var07"] ==0)):
             s.append("1033")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]>=-2.02413225 and data["var10"] !=0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var03"]<-2.21122217 or data["var03"] ==0) and (data["var38"]<1.05968142 or data["var38"] ==0)):
             s.append("1085")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]>=-2.02413225 and data["var10"] !=0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var03"]<-2.21122217 or data["var03"] ==0) and (data["var38"]>=1.05968142 and data["var38"] !=0)):
             s.append("1086")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]>=-2.02413225 and data["var10"] !=0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var03"]>=-2.21122217 and data["var03"] !=0) and (data["var51"]<1.60576987 or data["var51"] ==0)):
             s.append("1087")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]<2.31438637 or data["var37"] ==0) and (data["var10"]>=-2.02413225 and data["var10"] !=0) and (data["var07"]>=-2.46717882 and data["var07"] !=0) and (data["var03"]>=-2.21122217 and data["var03"] !=0) and (data["var51"]>=1.60576987 and data["var51"] !=0)):
             s.append("1088")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]>=2.31438637 and data["var37"] !=0) and (data["var60"]<-1.24750423 or data["var60"] ==0)):
             s.append("1017")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]>=2.31438637 and data["var37"] !=0) and (data["var60"]>=-1.24750423 and data["var60"] !=0) and (data["var06"]<2.14370108 or data["var06"] ==0)):
             s.append("1035")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]<1.86600971 or data["var36"] ==0) and (data["var37"]>=2.31438637 and data["var37"] !=0) and (data["var60"]>=-1.24750423 and data["var60"] !=0) and (data["var06"]>=2.14370108 and data["var06"] !=0)):
             s.append("1036")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]>=1.86600971 and data["var36"] !=0) and (data["var39"]<-0.92717433 or data["var39"] ==0) and (data["var21"]<-0.410057843 or data["var21"] ==0)):
             s.append("1019")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]>=1.86600971 and data["var36"] !=0) and (data["var39"]<-0.92717433 or data["var39"] ==0) and (data["var21"]>=-0.410057843 and data["var21"] !=0) and (data["var31"]<-0.512196004 or data["var31"] ==0)):
             s.append("1037")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]>=1.86600971 and data["var36"] !=0) and (data["var39"]<-0.92717433 or data["var39"] ==0) and (data["var21"]>=-0.410057843 and data["var21"] !=0) and (data["var31"]>=-0.512196004 and data["var31"] !=0)):
             s.append("1038")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]>=1.86600971 and data["var36"] !=0) and (data["var39"]>=-0.92717433 and data["var39"] !=0) and (data["var48"]<2.13045788 or data["var48"] ==0) and (data["var13"]<2.23584318 or data["var13"] ==0) and (data["var26"]<-1.93389893 or data["var26"] ==0)):
             s.append("1059")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]>=1.86600971 and data["var36"] !=0) and (data["var39"]>=-0.92717433 and data["var39"] !=0) and (data["var48"]<2.13045788 or data["var48"] ==0) and (data["var13"]<2.23584318 or data["var13"] ==0) and (data["var26"]>=-1.93389893 and data["var26"] !=0)):
             s.append("1060")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]>=1.86600971 and data["var36"] !=0) and (data["var39"]>=-0.92717433 and data["var39"] !=0) and (data["var48"]<2.13045788 or data["var48"] ==0) and (data["var13"]>=2.23584318 and data["var13"] !=0)):
             s.append("1040")
    if((data["var04"]<-1.96614754 or data["var04"] ==0) and (data["var36"]>=1.86600971 and data["var36"] !=0) and (data["var39"]>=-0.92717433 and data["var39"] !=0) and (data["var48"]>=2.13045788 and data["var48"] !=0)):
             s.append("1022")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]<1.27983701 or data["var16"] ==0) and (data["var12"]<-2.14346004 or data["var12"] ==0) and (data["var03"]<-1.2675494 or data["var03"] ==0)):
             s.append("1089")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]<1.27983701 or data["var16"] ==0) and (data["var12"]<-2.14346004 or data["var12"] ==0) and (data["var03"]>=-1.2675494 and data["var03"] !=0)):
             s.append("1090")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]<1.27983701 or data["var16"] ==0) and (data["var12"]>=-2.14346004 and data["var12"] !=0) and (data["var10"]<1.38066113 or data["var10"] ==0)):
             s.append("1091")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]<1.27983701 or data["var16"] ==0) and (data["var12"]>=-2.14346004 and data["var12"] !=0) and (data["var10"]>=1.38066113 and data["var10"] !=0)):
             s.append("1092")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]>=1.27983701 and data["var16"] !=0) and (data["var20"]<-0.0403215773 or data["var20"] ==0) and (data["var25"]<-1.13838303 or data["var25"] ==0)):
             s.append("1093")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]>=1.27983701 and data["var16"] !=0) and (data["var20"]<-0.0403215773 or data["var20"] ==0) and (data["var25"]>=-1.13838303 and data["var25"] !=0)):
             s.append("1094")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]>=1.27983701 and data["var16"] !=0) and (data["var20"]>=-0.0403215773 and data["var20"] !=0) and (data["var18"]<-0.830694199 or data["var18"] ==0)):
             s.append("1095")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]<2.3130331 or data["var07"] ==0) and (data["var16"]>=1.27983701 and data["var16"] !=0) and (data["var20"]>=-0.0403215773 and data["var20"] !=0) and (data["var18"]>=-0.830694199 and data["var18"] !=0)):
             s.append("1096")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]>=2.3130331 and data["var07"] !=0) and (data["var21"]<-2.12856936 or data["var21"] ==0)):
             s.append("1043")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]>=2.3130331 and data["var07"] !=0) and (data["var21"]>=-2.12856936 and data["var21"] !=0) and (data["var31"]<0.0852499977 or data["var31"] ==0) and (data["var59"]<-0.193041295 or data["var59"] ==0)):
             s.append("1097")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]>=2.3130331 and data["var07"] !=0) and (data["var21"]>=-2.12856936 and data["var21"] !=0) and (data["var31"]<0.0852499977 or data["var31"] ==0) and (data["var59"]>=-0.193041295 and data["var59"] !=0)):
             s.append("1098")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]<1.79720247 or data["var55"] ==0) and (data["var07"]>=2.3130331 and data["var07"] !=0) and (data["var21"]>=-2.12856936 and data["var21"] !=0) and (data["var31"]>=0.0852499977 and data["var31"] !=0)):
             s.append("1066")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]>=1.79720247 and data["var55"] !=0) and (data["var07"]<2.2939198 or data["var07"] ==0) and (data["var28"]<2.23098302 or data["var28"] ==0) and (data["var52"]<0.819534004 or data["var52"] ==0) and (data["var60"]<-2.06237793 or data["var60"] ==0)):
             s.append("1099")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]>=1.79720247 and data["var55"] !=0) and (data["var07"]<2.2939198 or data["var07"] ==0) and (data["var28"]<2.23098302 or data["var28"] ==0) and (data["var52"]<0.819534004 or data["var52"] ==0) and (data["var60"]>=-2.06237793 and data["var60"] !=0)):
             s.append("10100")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]>=1.79720247 and data["var55"] !=0) and (data["var07"]<2.2939198 or data["var07"] ==0) and (data["var28"]<2.23098302 or data["var28"] ==0) and (data["var52"]>=0.819534004 and data["var52"] !=0) and (data["var52"]<1.70241213 or data["var52"] ==0)):
             s.append("10101")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]>=1.79720247 and data["var55"] !=0) and (data["var07"]<2.2939198 or data["var07"] ==0) and (data["var28"]<2.23098302 or data["var28"] ==0) and (data["var52"]>=0.819534004 and data["var52"] !=0) and (data["var52"]>=1.70241213 and data["var52"] !=0)):
             s.append("10102")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]>=1.79720247 and data["var55"] !=0) and (data["var07"]<2.2939198 or data["var07"] ==0) and (data["var28"]>=2.23098302 and data["var28"] !=0)):
             s.append("1046")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]<-1.72923231 or data["var21"] ==0) and (data["var55"]>=1.79720247 and data["var55"] !=0) and (data["var07"]>=2.2939198 and data["var07"] !=0)):
             s.append("1026")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]<1.89987981 or data["var43"] ==0) and (data["var27"]<-2.03649569 or data["var27"] ==0) and (data["var26"]<1.07461643 or data["var26"] ==0)):
             s.append("10103")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]<1.89987981 or data["var43"] ==0) and (data["var27"]<-2.03649569 or data["var27"] ==0) and (data["var26"]>=1.07461643 and data["var26"] !=0)):
             s.append("10104")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]<1.89987981 or data["var43"] ==0) and (data["var27"]>=-2.03649569 and data["var27"] !=0) and (data["var03"]<2.30437779 or data["var03"] ==0)):
             s.append("10105")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]<1.89987981 or data["var43"] ==0) and (data["var27"]>=-2.03649569 and data["var27"] !=0) and (data["var03"]>=2.30437779 and data["var03"] !=0)):
             s.append("10106")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]>=1.89987981 and data["var43"] !=0) and (data["var14"]<1.11197448 or data["var14"] ==0) and (data["var05"]<-0.728044868 or data["var05"] ==0)):
             s.append("10107")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]>=1.89987981 and data["var43"] !=0) and (data["var14"]<1.11197448 or data["var14"] ==0) and (data["var05"]>=-0.728044868 and data["var05"] !=0)):
             s.append("10108")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]>=1.89987981 and data["var43"] !=0) and (data["var14"]>=1.11197448 and data["var14"] !=0) and (data["var45"]<-1.26830506 or data["var45"] ==0)):
             s.append("10109")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]<2.45246506 or data["var47"] ==0) and (data["var43"]>=1.89987981 and data["var43"] !=0) and (data["var14"]>=1.11197448 and data["var14"] !=0) and (data["var45"]>=-1.26830506 and data["var45"] !=0)):
             s.append("10110")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]>=2.45246506 and data["var47"] !=0) and (data["var56"]<-0.428641856 or data["var56"] ==0) and (data["var15"]<-1.79255462 or data["var15"] ==0)):
             s.append("1073")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]>=2.45246506 and data["var47"] !=0) and (data["var56"]<-0.428641856 or data["var56"] ==0) and (data["var15"]>=-1.79255462 and data["var15"] !=0) and (data["var25"]<1.14466405 or data["var25"] ==0)):
             s.append("10111")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]>=2.45246506 and data["var47"] !=0) and (data["var56"]<-0.428641856 or data["var56"] ==0) and (data["var15"]>=-1.79255462 and data["var15"] !=0) and (data["var25"]>=1.14466405 and data["var25"] !=0)):
             s.append("10112")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]>=2.45246506 and data["var47"] !=0) and (data["var56"]>=-0.428641856 and data["var56"] !=0) and (data["var41"]<1.4910326 or data["var41"] ==0) and (data["var45"]<-2.49230862 or data["var45"] ==0)):
             s.append("10113")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]>=2.45246506 and data["var47"] !=0) and (data["var56"]>=-0.428641856 and data["var56"] !=0) and (data["var41"]<1.4910326 or data["var41"] ==0) and (data["var45"]>=-2.49230862 and data["var45"] !=0)):
             s.append("10114")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]>=2.45246506 and data["var47"] !=0) and (data["var56"]>=-0.428641856 and data["var56"] !=0) and (data["var41"]>=1.4910326 and data["var41"] !=0) and (data["var42"]<0.108542591 or data["var42"] ==0)):
             s.append("10115")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]<2.23053741 or data["var33"] ==0) and (data["var47"]>=2.45246506 and data["var47"] !=0) and (data["var56"]>=-0.428641856 and data["var56"] !=0) and (data["var41"]>=1.4910326 and data["var41"] !=0) and (data["var42"]>=0.108542591 and data["var42"] !=0)):
             s.append("10116")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]<0.851965606 or data["var60"] ==0) and (data["var60"]<0.571883023 or data["var60"] ==0) and (data["var43"]<-0.952571571 or data["var43"] ==0) and (data["var59"]<0.356301606 or data["var59"] ==0)):
             s.append("10117")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]<0.851965606 or data["var60"] ==0) and (data["var60"]<0.571883023 or data["var60"] ==0) and (data["var43"]<-0.952571571 or data["var43"] ==0) and (data["var59"]>=0.356301606 and data["var59"] !=0)):
             s.append("10118")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]<0.851965606 or data["var60"] ==0) and (data["var60"]<0.571883023 or data["var60"] ==0) and (data["var43"]>=-0.952571571 and data["var43"] !=0) and (data["var25"]<-1.84111285 or data["var25"] ==0)):
             s.append("10119")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]<0.851965606 or data["var60"] ==0) and (data["var60"]<0.571883023 or data["var60"] ==0) and (data["var43"]>=-0.952571571 and data["var43"] !=0) and (data["var25"]>=-1.84111285 and data["var25"] !=0)):
             s.append("10120")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]<0.851965606 or data["var60"] ==0) and (data["var60"]>=0.571883023 and data["var60"] !=0) and (data["var37"]<-1.37687182 or data["var37"] ==0)):
             s.append("1079")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]<0.851965606 or data["var60"] ==0) and (data["var60"]>=0.571883023 and data["var60"] !=0) and (data["var37"]>=-1.37687182 and data["var37"] !=0) and (data["var57"]<1.68827176 or data["var57"] ==0)):
             s.append("10121")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]<0.851965606 or data["var60"] ==0) and (data["var60"]>=0.571883023 and data["var60"] !=0) and (data["var37"]>=-1.37687182 and data["var37"] !=0) and (data["var57"]>=1.68827176 and data["var57"] !=0)):
             s.append("10122")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]>=0.851965606 and data["var60"] !=0) and (data["var24"]<-0.0686432868 or data["var24"] ==0) and (data["var04"]<2.16580629 or data["var04"] ==0) and (data["var52"]<-0.688556969 or data["var52"] ==0)):
             s.append("10123")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]>=0.851965606 and data["var60"] !=0) and (data["var24"]<-0.0686432868 or data["var24"] ==0) and (data["var04"]<2.16580629 or data["var04"] ==0) and (data["var52"]>=-0.688556969 and data["var52"] !=0)):
             s.append("10124")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]>=0.851965606 and data["var60"] !=0) and (data["var24"]<-0.0686432868 or data["var24"] ==0) and (data["var04"]>=2.16580629 and data["var04"] !=0)):
             s.append("1082")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]>=0.851965606 and data["var60"] !=0) and (data["var24"]>=-0.0686432868 and data["var24"] !=0) and (data["var55"]<1.68128061 or data["var55"] ==0) and (data["var35"]<1.39238214 or data["var35"] ==0)):
             s.append("10125")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]>=0.851965606 and data["var60"] !=0) and (data["var24"]>=-0.0686432868 and data["var24"] !=0) and (data["var55"]<1.68128061 or data["var55"] ==0) and (data["var35"]>=1.39238214 and data["var35"] !=0)):
             s.append("10126")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]>=0.851965606 and data["var60"] !=0) and (data["var24"]>=-0.0686432868 and data["var24"] !=0) and (data["var55"]>=1.68128061 and data["var55"] !=0) and (data["var44"]<-0.105551496 or data["var44"] ==0)):
             s.append("10127")
    if((data["var04"]>=-1.96614754 and data["var04"] !=0) and (data["var21"]>=-1.72923231 and data["var21"] !=0) and (data["var33"]>=2.23053741 and data["var33"] !=0) and (data["var60"]>=0.851965606 and data["var60"] !=0) and (data["var24"]>=-0.0686432868 and data["var24"] !=0) and (data["var55"]>=1.68128061 and data["var55"] !=0) and (data["var44"]>=-0.105551496 and data["var44"] !=0)):
             s.append("10128")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]<-1.33887362 or data["var52"] ==0) and (data["var21"]<0.879513383 or data["var21"] ==0) and (data["var24"]<0.100825563 or data["var24"] ==0) and (data["var29"]<-0.0863725767 or data["var29"] ==0)):
             s.append("1195")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]<-1.33887362 or data["var52"] ==0) and (data["var21"]<0.879513383 or data["var21"] ==0) and (data["var24"]<0.100825563 or data["var24"] ==0) and (data["var29"]>=-0.0863725767 and data["var29"] !=0)):
             s.append("1196")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]<-1.33887362 or data["var52"] ==0) and (data["var21"]<0.879513383 or data["var21"] ==0) and (data["var24"]>=0.100825563 and data["var24"] !=0) and (data["var25"]<1.07303452 or data["var25"] ==0)):
             s.append("1197")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]<-1.33887362 or data["var52"] ==0) and (data["var21"]<0.879513383 or data["var21"] ==0) and (data["var24"]>=0.100825563 and data["var24"] !=0) and (data["var25"]>=1.07303452 and data["var25"] !=0)):
             s.append("1198")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]<-1.33887362 or data["var52"] ==0) and (data["var21"]>=0.879513383 and data["var21"] !=0)):
             s.append("1132")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]>=-1.33887362 and data["var52"] !=0) and (data["var58"]<2.39795542 or data["var58"] ==0) and (data["var58"]<-0.786747098 or data["var58"] ==0) and (data["var12"]<1.40064597 or data["var12"] ==0)):
             s.append("1199")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]>=-1.33887362 and data["var52"] !=0) and (data["var58"]<2.39795542 or data["var58"] ==0) and (data["var58"]<-0.786747098 or data["var58"] ==0) and (data["var12"]>=1.40064597 and data["var12"] !=0)):
             s.append("11100")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]>=-1.33887362 and data["var52"] !=0) and (data["var58"]<2.39795542 or data["var58"] ==0) and (data["var58"]>=-0.786747098 and data["var58"] !=0) and (data["var25"]<2.60500813 or data["var25"] ==0)):
             s.append("11101")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]>=-1.33887362 and data["var52"] !=0) and (data["var58"]<2.39795542 or data["var58"] ==0) and (data["var58"]>=-0.786747098 and data["var58"] !=0) and (data["var25"]>=2.60500813 and data["var25"] !=0)):
             s.append("11102")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]>=-1.33887362 and data["var52"] !=0) and (data["var58"]>=2.39795542 and data["var58"] !=0) and (data["var27"]<-0.538619459 or data["var27"] ==0)):
             s.append("1163")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]<1.21303308 or data["var50"] ==0) and (data["var52"]>=-1.33887362 and data["var52"] !=0) and (data["var58"]>=2.39795542 and data["var58"] !=0) and (data["var27"]>=-0.538619459 and data["var27"] !=0)):
             s.append("1164")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]<-1.3881098 or data["var39"] ==0) and (data["var16"]<-0.406352937 or data["var16"] ==0)):
             s.append("1135")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]<-1.3881098 or data["var39"] ==0) and (data["var16"]>=-0.406352937 and data["var16"] !=0) and (data["var37"]<-0.381311774 or data["var37"] ==0)):
             s.append("1165")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]<-1.3881098 or data["var39"] ==0) and (data["var16"]>=-0.406352937 and data["var16"] !=0) and (data["var37"]>=-0.381311774 and data["var37"] !=0)):
             s.append("1166")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]>=-1.3881098 and data["var39"] !=0) and (data["var60"]<-0.269550443 or data["var60"] ==0) and (data["var05"]<1.58812988 or data["var05"] ==0) and (data["var24"]<-1.99868369 or data["var24"] ==0)):
             s.append("11103")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]>=-1.3881098 and data["var39"] !=0) and (data["var60"]<-0.269550443 or data["var60"] ==0) and (data["var05"]<1.58812988 or data["var05"] ==0) and (data["var24"]>=-1.99868369 and data["var24"] !=0)):
             s.append("11104")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]>=-1.3881098 and data["var39"] !=0) and (data["var60"]<-0.269550443 or data["var60"] ==0) and (data["var05"]>=1.58812988 and data["var05"] !=0)):
             s.append("1168")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]>=-1.3881098 and data["var39"] !=0) and (data["var60"]>=-0.269550443 and data["var60"] !=0) and (data["var60"]<0.262921691 or data["var60"] ==0) and (data["var42"]<0.488451213 or data["var42"] ==0)):
             s.append("11105")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]>=-1.3881098 and data["var39"] !=0) and (data["var60"]>=-0.269550443 and data["var60"] !=0) and (data["var60"]<0.262921691 or data["var60"] ==0) and (data["var42"]>=0.488451213 and data["var42"] !=0)):
             s.append("11106")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]>=-1.3881098 and data["var39"] !=0) and (data["var60"]>=-0.269550443 and data["var60"] !=0) and (data["var60"]>=0.262921691 and data["var60"] !=0) and (data["var16"]<-2.12557912 or data["var16"] ==0)):
             s.append("11107")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]<1.60324311 or data["var60"] ==0) and (data["var50"]>=1.21303308 and data["var50"] !=0) and (data["var39"]>=-1.3881098 and data["var39"] !=0) and (data["var60"]>=-0.269550443 and data["var60"] !=0) and (data["var60"]>=0.262921691 and data["var60"] !=0) and (data["var16"]>=-2.12557912 and data["var16"] !=0)):
             s.append("11108")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]<-0.811827302 or data["var34"] ==0) and (data["var58"]<-0.744901776 or data["var58"] ==0) and (data["var40"]<-0.359147012 or data["var40"] ==0)):
             s.append("1139")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]<-0.811827302 or data["var34"] ==0) and (data["var58"]<-0.744901776 or data["var58"] ==0) and (data["var40"]>=-0.359147012 and data["var40"] !=0)):
             s.append("1140")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]<-0.811827302 or data["var34"] ==0) and (data["var58"]>=-0.744901776 and data["var58"] !=0) and (data["var45"]<0.163626313 or data["var45"] ==0) and (data["var06"]<-0.504081666 or data["var06"] ==0)):
             s.append("1171")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]<-0.811827302 or data["var34"] ==0) and (data["var58"]>=-0.744901776 and data["var58"] !=0) and (data["var45"]<0.163626313 or data["var45"] ==0) and (data["var06"]>=-0.504081666 and data["var06"] !=0)):
             s.append("1172")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]<-0.811827302 or data["var34"] ==0) and (data["var58"]>=-0.744901776 and data["var58"] !=0) and (data["var45"]>=0.163626313 and data["var45"] !=0)):
             s.append("1142")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]>=-0.811827302 and data["var34"] !=0) and (data["var45"]<1.57585883 or data["var45"] ==0) and (data["var10"]<-1.69310045 or data["var10"] ==0)):
             s.append("1143")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]>=-0.811827302 and data["var34"] !=0) and (data["var45"]<1.57585883 or data["var45"] ==0) and (data["var10"]>=-1.69310045 and data["var10"] !=0) and (data["var49"]<-1.48479748 or data["var49"] ==0)):
             s.append("1173")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]>=-0.811827302 and data["var34"] !=0) and (data["var45"]<1.57585883 or data["var45"] ==0) and (data["var10"]>=-1.69310045 and data["var10"] !=0) and (data["var49"]>=-1.48479748 and data["var49"] !=0) and (data["var07"]<-1.73848271 or data["var07"] ==0)):
             s.append("11109")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]>=-0.811827302 and data["var34"] !=0) and (data["var45"]<1.57585883 or data["var45"] ==0) and (data["var10"]>=-1.69310045 and data["var10"] !=0) and (data["var49"]>=-1.48479748 and data["var49"] !=0) and (data["var07"]>=-1.73848271 and data["var07"] !=0)):
             s.append("11110")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]>=-0.811827302 and data["var34"] !=0) and (data["var45"]>=1.57585883 and data["var45"] !=0) and (data["var40"]<0.586584926 or data["var40"] ==0)):
             s.append("1145")
    if((data["var03"]<-1.91103721 or data["var03"] ==0) and (data["var60"]>=1.60324311 and data["var60"] !=0) and (data["var34"]>=-0.811827302 and data["var34"] !=0) and (data["var45"]>=1.57585883 and data["var45"] !=0) and (data["var40"]>=0.586584926 and data["var40"] !=0)):
             s.append("1146")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]<0.947481155 or data["var09"] ==0) and (data["var44"]<0.736663222 or data["var44"] ==0) and (data["var24"]<1.12321043 or data["var24"] ==0)):
             s.append("11111")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]<0.947481155 or data["var09"] ==0) and (data["var44"]<0.736663222 or data["var44"] ==0) and (data["var24"]>=1.12321043 and data["var24"] !=0)):
             s.append("11112")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]<0.947481155 or data["var09"] ==0) and (data["var44"]>=0.736663222 and data["var44"] !=0) and (data["var48"]<0.833841264 or data["var48"] ==0)):
             s.append("11113")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]<0.947481155 or data["var09"] ==0) and (data["var44"]>=0.736663222 and data["var44"] !=0) and (data["var48"]>=0.833841264 and data["var48"] !=0)):
             s.append("11114")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]>=0.947481155 and data["var09"] !=0) and (data["var12"]<-2.85006618 or data["var12"] ==0) and (data["var24"]<-1.34938049 or data["var24"] ==0)):
             s.append("11115")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]>=0.947481155 and data["var09"] !=0) and (data["var12"]<-2.85006618 or data["var12"] ==0) and (data["var24"]>=-1.34938049 and data["var24"] !=0)):
             s.append("11116")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]>=0.947481155 and data["var09"] !=0) and (data["var12"]>=-2.85006618 and data["var12"] !=0) and (data["var32"]<0.524379432 or data["var32"] ==0)):
             s.append("11117")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]<-2.06638384 or data["var12"] ==0) and (data["var09"]>=0.947481155 and data["var09"] !=0) and (data["var12"]>=-2.85006618 and data["var12"] !=0) and (data["var32"]>=0.524379432 and data["var32"] !=0)):
             s.append("11118")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]<2.81464338 or data["var38"] ==0) and (data["var11"]<-1.62727773 or data["var11"] ==0) and (data["var11"]<-3.27959967 or data["var11"] ==0)):
             s.append("11119")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]<2.81464338 or data["var38"] ==0) and (data["var11"]<-1.62727773 or data["var11"] ==0) and (data["var11"]>=-3.27959967 and data["var11"] !=0)):
             s.append("11120")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]<2.81464338 or data["var38"] ==0) and (data["var11"]>=-1.62727773 and data["var11"] !=0) and (data["var29"]<-2.21378732 or data["var29"] ==0)):
             s.append("11121")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]<2.81464338 or data["var38"] ==0) and (data["var11"]>=-1.62727773 and data["var11"] !=0) and (data["var29"]>=-2.21378732 and data["var29"] !=0)):
             s.append("11122")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]>=2.81464338 and data["var38"] !=0) and (data["var28"]<0.381589532 or data["var28"] ==0) and (data["var19"]<1.40782285 or data["var19"] ==0)):
             s.append("11123")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]>=2.81464338 and data["var38"] !=0) and (data["var28"]<0.381589532 or data["var28"] ==0) and (data["var19"]>=1.40782285 and data["var19"] !=0)):
             s.append("11124")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]>=2.81464338 and data["var38"] !=0) and (data["var28"]>=0.381589532 and data["var28"] !=0) and (data["var40"]<-0.29018873 or data["var40"] ==0)):
             s.append("11125")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]<2.23157835 or data["var36"] ==0) and (data["var12"]>=-2.06638384 and data["var12"] !=0) and (data["var38"]>=2.81464338 and data["var38"] !=0) and (data["var28"]>=0.381589532 and data["var28"] !=0) and (data["var40"]>=-0.29018873 and data["var40"] !=0)):
             s.append("11126")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]<-1.37722421 or data["var21"] ==0) and (data["var53"]<-0.00208922289 or data["var53"] ==0) and (data["var39"]<1.65148425 or data["var39"] ==0)):
             s.append("1183")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]<-1.37722421 or data["var21"] ==0) and (data["var53"]<-0.00208922289 or data["var53"] ==0) and (data["var39"]>=1.65148425 and data["var39"] !=0)):
             s.append("1184")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]<-1.37722421 or data["var21"] ==0) and (data["var53"]>=-0.00208922289 and data["var53"] !=0) and (data["var51"]<-1.17147636 or data["var51"] ==0) and (data["var26"]<0.297940463 or data["var26"] ==0)):
             s.append("11127")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]<-1.37722421 or data["var21"] ==0) and (data["var53"]>=-0.00208922289 and data["var53"] !=0) and (data["var51"]<-1.17147636 or data["var51"] ==0) and (data["var26"]>=0.297940463 and data["var26"] !=0)):
             s.append("11128")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]<-1.37722421 or data["var21"] ==0) and (data["var53"]>=-0.00208922289 and data["var53"] !=0) and (data["var51"]>=-1.17147636 and data["var51"] !=0) and (data["var10"]<-0.816587925 or data["var10"] ==0)):
             s.append("11129")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]<-1.37722421 or data["var21"] ==0) and (data["var53"]>=-0.00208922289 and data["var53"] !=0) and (data["var51"]>=-1.17147636 and data["var51"] !=0) and (data["var10"]>=-0.816587925 and data["var10"] !=0)):
             s.append("11130")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]<0.309218228 or data["var12"] ==0) and (data["var07"]<-0.274846017 or data["var07"] ==0) and (data["var27"]<-0.466163576 or data["var27"] ==0)):
             s.append("11131")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]<0.309218228 or data["var12"] ==0) and (data["var07"]<-0.274846017 or data["var07"] ==0) and (data["var27"]>=-0.466163576 and data["var27"] !=0)):
             s.append("11132")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]<0.309218228 or data["var12"] ==0) and (data["var07"]>=-0.274846017 and data["var07"] !=0) and (data["var28"]<-2.15881586 or data["var28"] ==0)):
             s.append("11133")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]<0.309218228 or data["var12"] ==0) and (data["var07"]>=-0.274846017 and data["var07"] !=0) and (data["var28"]>=-2.15881586 and data["var28"] !=0)):
             s.append("11134")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]>=0.309218228 and data["var12"] !=0) and (data["var31"]<0.971021295 or data["var31"] ==0) and (data["var20"]<0.271640211 or data["var20"] ==0)):
             s.append("11135")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]>=0.309218228 and data["var12"] !=0) and (data["var31"]<0.971021295 or data["var31"] ==0) and (data["var20"]>=0.271640211 and data["var20"] !=0)):
             s.append("11136")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]>=0.309218228 and data["var12"] !=0) and (data["var31"]>=0.971021295 and data["var31"] !=0) and (data["var25"]<-1.72430682 or data["var25"] ==0)):
             s.append("11137")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]<2.90511894 or data["var55"] ==0) and (data["var36"]>=2.23157835 and data["var36"] !=0) and (data["var21"]>=-1.37722421 and data["var21"] !=0) and (data["var12"]>=0.309218228 and data["var12"] !=0) and (data["var31"]>=0.971021295 and data["var31"] !=0) and (data["var25"]>=-1.72430682 and data["var25"] !=0)):
             s.append("11138")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]<0.387119114 or data["var08"] ==0) and (data["var06"]<2.22999382 or data["var06"] ==0) and (data["var19"]<1.76236129 or data["var19"] ==0) and (data["var48"]<2.22842097 or data["var48"] ==0) and (data["var59"]<1.08507991 or data["var59"] ==0)):
             s.append("11139")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]<0.387119114 or data["var08"] ==0) and (data["var06"]<2.22999382 or data["var06"] ==0) and (data["var19"]<1.76236129 or data["var19"] ==0) and (data["var48"]<2.22842097 or data["var48"] ==0) and (data["var59"]>=1.08507991 and data["var59"] !=0)):
             s.append("11140")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]<0.387119114 or data["var08"] ==0) and (data["var06"]<2.22999382 or data["var06"] ==0) and (data["var19"]<1.76236129 or data["var19"] ==0) and (data["var48"]>=2.22842097 and data["var48"] !=0)):
             s.append("1192")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]<0.387119114 or data["var08"] ==0) and (data["var06"]<2.22999382 or data["var06"] ==0) and (data["var19"]>=1.76236129 and data["var19"] !=0)):
             s.append("1156")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]<0.387119114 or data["var08"] ==0) and (data["var06"]>=2.22999382 and data["var06"] !=0)):
             s.append("1128")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]>=0.387119114 and data["var08"] !=0) and (data["var11"]<1.0775075 or data["var11"] ==0) and (data["var20"]<-1.04132187 or data["var20"] ==0)):
             s.append("1157")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]>=0.387119114 and data["var08"] !=0) and (data["var11"]<1.0775075 or data["var11"] ==0) and (data["var20"]>=-1.04132187 and data["var20"] !=0) and (data["var12"]<-0.611722827 or data["var12"] ==0) and (data["var46"]<0.545303702 or data["var46"] ==0)):
             s.append("11141")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]>=0.387119114 and data["var08"] !=0) and (data["var11"]<1.0775075 or data["var11"] ==0) and (data["var20"]>=-1.04132187 and data["var20"] !=0) and (data["var12"]<-0.611722827 or data["var12"] ==0) and (data["var46"]>=0.545303702 and data["var46"] !=0)):
             s.append("11142")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]>=0.387119114 and data["var08"] !=0) and (data["var11"]<1.0775075 or data["var11"] ==0) and (data["var20"]>=-1.04132187 and data["var20"] !=0) and (data["var12"]>=-0.611722827 and data["var12"] !=0) and (data["var24"]<-0.554720044 or data["var24"] ==0)):
             s.append("11143")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]>=0.387119114 and data["var08"] !=0) and (data["var11"]<1.0775075 or data["var11"] ==0) and (data["var20"]>=-1.04132187 and data["var20"] !=0) and (data["var12"]>=-0.611722827 and data["var12"] !=0) and (data["var24"]>=-0.554720044 and data["var24"] !=0)):
             s.append("11144")
    if((data["var03"]>=-1.91103721 and data["var03"] !=0) and (data["var55"]>=2.90511894 and data["var55"] !=0) and (data["var08"]>=0.387119114 and data["var08"] !=0) and (data["var11"]>=1.0775075 and data["var11"] !=0)):
             s.append("1130")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]<-0.793582082 or data["var21"] ==0) and (data["var16"]<-0.313148141 or data["var16"] ==0)):
             s.append("1215")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]<-0.793582082 or data["var21"] ==0) and (data["var16"]>=-0.313148141 and data["var16"] !=0) and (data["var35"]<-0.330076635 or data["var35"] ==0) and (data["var16"]<-0.216878101 or data["var16"] ==0)):
             s.append("1257")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]<-0.793582082 or data["var21"] ==0) and (data["var16"]>=-0.313148141 and data["var16"] !=0) and (data["var35"]<-0.330076635 or data["var35"] ==0) and (data["var16"]>=-0.216878101 and data["var16"] !=0)):
             s.append("1258")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]<-0.793582082 or data["var21"] ==0) and (data["var16"]>=-0.313148141 and data["var16"] !=0) and (data["var35"]>=-0.330076635 and data["var35"] !=0) and (data["var08"]<0.0403442495 or data["var08"] ==0) and (data["var51"]<1.24959707 or data["var51"] ==0)):
             s.append("12101")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]<-0.793582082 or data["var21"] ==0) and (data["var16"]>=-0.313148141 and data["var16"] !=0) and (data["var35"]>=-0.330076635 and data["var35"] !=0) and (data["var08"]<0.0403442495 or data["var08"] ==0) and (data["var51"]>=1.24959707 and data["var51"] !=0)):
             s.append("12102")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]<-0.793582082 or data["var21"] ==0) and (data["var16"]>=-0.313148141 and data["var16"] !=0) and (data["var35"]>=-0.330076635 and data["var35"] !=0) and (data["var08"]>=0.0403442495 and data["var08"] !=0) and (data["var36"]<1.19090366 or data["var36"] ==0)):
             s.append("12103")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]<-0.793582082 or data["var21"] ==0) and (data["var16"]>=-0.313148141 and data["var16"] !=0) and (data["var35"]>=-0.330076635 and data["var35"] !=0) and (data["var08"]>=0.0403442495 and data["var08"] !=0) and (data["var36"]>=1.19090366 and data["var36"] !=0)):
             s.append("12104")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]<-1.22265077 or data["var34"] ==0) and (data["var01"]<0.376065016 or data["var01"] ==0) and (data["var30"]<1.12573743 or data["var30"] ==0)):
             s.append("1261")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]<-1.22265077 or data["var34"] ==0) and (data["var01"]<0.376065016 or data["var01"] ==0) and (data["var30"]>=1.12573743 and data["var30"] !=0)):
             s.append("1262")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]<-1.22265077 or data["var34"] ==0) and (data["var01"]>=0.376065016 and data["var01"] !=0) and (data["var38"]<-0.140636384 or data["var38"] ==0)):
             s.append("1263")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]<-1.22265077 or data["var34"] ==0) and (data["var01"]>=0.376065016 and data["var01"] !=0) and (data["var38"]>=-0.140636384 and data["var38"] !=0)):
             s.append("1264")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]>=-1.22265077 and data["var34"] !=0) and (data["var14"]<-1.22842002 or data["var14"] ==0) and (data["var51"]<-1.58567417 or data["var51"] ==0)):
             s.append("1265")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]>=-1.22265077 and data["var34"] !=0) and (data["var14"]<-1.22842002 or data["var14"] ==0) and (data["var51"]>=-1.58567417 and data["var51"] !=0) and (data["var31"]<-1.4440217 or data["var31"] ==0)):
             s.append("12105")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]>=-1.22265077 and data["var34"] !=0) and (data["var14"]<-1.22842002 or data["var14"] ==0) and (data["var51"]>=-1.58567417 and data["var51"] !=0) and (data["var31"]>=-1.4440217 and data["var31"] !=0)):
             s.append("12106")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]>=-1.22265077 and data["var34"] !=0) and (data["var14"]>=-1.22842002 and data["var14"] !=0) and (data["var56"]<0.756559134 or data["var56"] ==0) and (data["var56"]<-2.09937 or data["var56"] ==0)):
             s.append("12107")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]>=-1.22265077 and data["var34"] !=0) and (data["var14"]>=-1.22842002 and data["var14"] !=0) and (data["var56"]<0.756559134 or data["var56"] ==0) and (data["var56"]>=-2.09937 and data["var56"] !=0)):
             s.append("12108")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]>=-1.22265077 and data["var34"] !=0) and (data["var14"]>=-1.22842002 and data["var14"] !=0) and (data["var56"]>=0.756559134 and data["var56"] !=0) and (data["var50"]<-0.708376169 or data["var50"] ==0)):
             s.append("12109")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]<-2.4005003 or data["var15"] ==0) and (data["var21"]>=-0.793582082 and data["var21"] !=0) and (data["var34"]>=-1.22265077 and data["var34"] !=0) and (data["var14"]>=-1.22842002 and data["var14"] !=0) and (data["var56"]>=0.756559134 and data["var56"] !=0) and (data["var50"]>=-0.708376169 and data["var50"] !=0)):
             s.append("12110")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]<-1.73877382 or data["var04"] ==0) and (data["var37"]<2.70770216 or data["var37"] ==0) and (data["var38"]<1.5559895 or data["var38"] ==0) and (data["var58"]<-2.26722383 or data["var58"] ==0)):
             s.append("12111")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]<-1.73877382 or data["var04"] ==0) and (data["var37"]<2.70770216 or data["var37"] ==0) and (data["var38"]<1.5559895 or data["var38"] ==0) and (data["var58"]>=-2.26722383 and data["var58"] !=0)):
             s.append("12112")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]<-1.73877382 or data["var04"] ==0) and (data["var37"]<2.70770216 or data["var37"] ==0) and (data["var38"]>=1.5559895 and data["var38"] !=0) and (data["var39"]<-0.448542237 or data["var39"] ==0)):
             s.append("12113")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]<-1.73877382 or data["var04"] ==0) and (data["var37"]<2.70770216 or data["var37"] ==0) and (data["var38"]>=1.5559895 and data["var38"] !=0) and (data["var39"]>=-0.448542237 and data["var39"] !=0)):
             s.append("12114")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]<-1.73877382 or data["var04"] ==0) and (data["var37"]>=2.70770216 and data["var37"] !=0)):
             s.append("1238")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]<-2.89970303 or data["var26"] ==0) and (data["var15"]<-0.0218676329 or data["var15"] ==0) and (data["var49"]<-0.73655045 or data["var49"] ==0)):
             s.append("12115")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]<-2.89970303 or data["var26"] ==0) and (data["var15"]<-0.0218676329 or data["var15"] ==0) and (data["var49"]>=-0.73655045 and data["var49"] !=0)):
             s.append("12116")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]<-2.89970303 or data["var26"] ==0) and (data["var15"]>=-0.0218676329 and data["var15"] !=0) and (data["var32"]<0.435611725 or data["var32"] ==0)):
             s.append("12117")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]<-2.89970303 or data["var26"] ==0) and (data["var15"]>=-0.0218676329 and data["var15"] !=0) and (data["var32"]>=0.435611725 and data["var32"] !=0)):
             s.append("12118")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]>=-2.89970303 and data["var26"] !=0) and (data["var38"]<-2.02761459 or data["var38"] ==0) and (data["var01"]<1.88063419 or data["var01"] ==0)):
             s.append("12119")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]>=-2.89970303 and data["var26"] !=0) and (data["var38"]<-2.02761459 or data["var38"] ==0) and (data["var01"]>=1.88063419 and data["var01"] !=0)):
             s.append("12120")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]>=-2.89970303 and data["var26"] !=0) and (data["var38"]>=-2.02761459 and data["var38"] !=0) and (data["var40"]<-2.15316963 or data["var40"] ==0)):
             s.append("12121")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]<1.35565162 or data["var15"] ==0) and (data["var04"]>=-1.73877382 and data["var04"] !=0) and (data["var26"]>=-2.89970303 and data["var26"] !=0) and (data["var38"]>=-2.02761459 and data["var38"] !=0) and (data["var40"]>=-2.15316963 and data["var40"] !=0)):
             s.append("12122")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]<-0.314687967 or data["var44"] ==0) and (data["var16"]<-0.31514284 or data["var16"] ==0) and (data["var33"]<-1.24025726 or data["var33"] ==0)):
             s.append("12123")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]<-0.314687967 or data["var44"] ==0) and (data["var16"]<-0.31514284 or data["var16"] ==0) and (data["var33"]>=-1.24025726 and data["var33"] !=0)):
             s.append("12124")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]<-0.314687967 or data["var44"] ==0) and (data["var16"]>=-0.31514284 and data["var16"] !=0) and (data["var60"]<0.65231961 or data["var60"] ==0)):
             s.append("12125")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]<-0.314687967 or data["var44"] ==0) and (data["var16"]>=-0.31514284 and data["var16"] !=0) and (data["var60"]>=0.65231961 and data["var60"] !=0)):
             s.append("12126")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]>=-0.314687967 and data["var44"] !=0) and (data["var38"]<1.73695552 or data["var38"] ==0) and (data["var50"]<-1.51288939 or data["var50"] ==0)):
             s.append("12127")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]>=-0.314687967 and data["var44"] !=0) and (data["var38"]<1.73695552 or data["var38"] ==0) and (data["var50"]>=-1.51288939 and data["var50"] !=0)):
             s.append("12128")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]>=-0.314687967 and data["var44"] !=0) and (data["var38"]>=1.73695552 and data["var38"] !=0) and (data["var03"]<0.810380578 or data["var03"] ==0)):
             s.append("12129")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]<-1.49432206 or data["var39"] ==0) and (data["var44"]>=-0.314687967 and data["var44"] !=0) and (data["var38"]>=1.73695552 and data["var38"] !=0) and (data["var03"]>=0.810380578 and data["var03"] !=0)):
             s.append("12130")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]<2.62275553 or data["var49"] ==0) and (data["var32"]<1.31056643 or data["var32"] ==0) and (data["var23"]<1.8770324 or data["var23"] ==0)):
             s.append("12131")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]<2.62275553 or data["var49"] ==0) and (data["var32"]<1.31056643 or data["var32"] ==0) and (data["var23"]>=1.8770324 and data["var23"] !=0)):
             s.append("12132")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]<2.62275553 or data["var49"] ==0) and (data["var32"]>=1.31056643 and data["var32"] !=0) and (data["var18"]<1.46070123 or data["var18"] ==0)):
             s.append("12133")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]<2.62275553 or data["var49"] ==0) and (data["var32"]>=1.31056643 and data["var32"] !=0) and (data["var18"]>=1.46070123 and data["var18"] !=0)):
             s.append("12134")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]>=2.62275553 and data["var49"] !=0) and (data["var60"]<1.42937565 or data["var60"] ==0) and (data["var03"]<-1.73950601 or data["var03"] ==0)):
             s.append("12135")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]>=2.62275553 and data["var49"] !=0) and (data["var60"]<1.42937565 or data["var60"] ==0) and (data["var03"]>=-1.73950601 and data["var03"] !=0)):
             s.append("12136")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]>=2.62275553 and data["var49"] !=0) and (data["var60"]>=1.42937565 and data["var60"] !=0) and (data["var43"]<-0.284823298 or data["var43"] ==0)):
             s.append("12137")
    if((data["var39"]<2.54962015 or data["var39"] ==0) and (data["var15"]>=-2.4005003 and data["var15"] !=0) and (data["var15"]>=1.35565162 and data["var15"] !=0) and (data["var39"]>=-1.49432206 and data["var39"] !=0) and (data["var49"]>=2.62275553 and data["var49"] !=0) and (data["var60"]>=1.42937565 and data["var60"] !=0) and (data["var43"]>=-0.284823298 and data["var43"] !=0)):
             s.append("12138")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]<-1.24841738 or data["var37"] ==0) and (data["var03"]<0.370500267 or data["var03"] ==0)):
             s.append("1245")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]<-1.24841738 or data["var37"] ==0) and (data["var03"]>=0.370500267 and data["var03"] !=0)):
             s.append("1246")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]>=-1.24841738 and data["var37"] !=0) and (data["var60"]<0.651380301 or data["var60"] ==0) and (data["var29"]<-1.76533473 or data["var29"] ==0)):
             s.append("1283")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]>=-1.24841738 and data["var37"] !=0) and (data["var60"]<0.651380301 or data["var60"] ==0) and (data["var29"]>=-1.76533473 and data["var29"] !=0) and (data["var27"]<-2.19475603 or data["var27"] ==0)):
             s.append("12139")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]>=-1.24841738 and data["var37"] !=0) and (data["var60"]<0.651380301 or data["var60"] ==0) and (data["var29"]>=-1.76533473 and data["var29"] !=0) and (data["var27"]>=-2.19475603 and data["var27"] !=0)):
             s.append("12140")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]>=-1.24841738 and data["var37"] !=0) and (data["var60"]>=0.651380301 and data["var60"] !=0) and (data["var03"]<-1.03071535 or data["var03"] ==0)):
             s.append("1285")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]>=-1.24841738 and data["var37"] !=0) and (data["var60"]>=0.651380301 and data["var60"] !=0) and (data["var03"]>=-1.03071535 and data["var03"] !=0) and (data["var21"]<0.504487395 or data["var21"] ==0)):
             s.append("12141")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]<-0.495757073 or data["var49"] ==0) and (data["var37"]>=-1.24841738 and data["var37"] !=0) and (data["var60"]>=0.651380301 and data["var60"] !=0) and (data["var03"]>=-1.03071535 and data["var03"] !=0) and (data["var21"]>=0.504487395 and data["var21"] !=0)):
             s.append("12142")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]<1.29275966 or data["var24"] ==0) and (data["var31"]<-1.33232117 or data["var31"] ==0) and (data["var50"]<0.479474485 or data["var50"] ==0)):
             s.append("1287")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]<1.29275966 or data["var24"] ==0) and (data["var31"]<-1.33232117 or data["var31"] ==0) and (data["var50"]>=0.479474485 and data["var50"] !=0)):
             s.append("1288")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]<1.29275966 or data["var24"] ==0) and (data["var31"]>=-1.33232117 and data["var31"] !=0) and (data["var57"]<-1.15386295 or data["var57"] ==0) and (data["var13"]<-0.496492088 or data["var13"] ==0)):
             s.append("12143")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]<1.29275966 or data["var24"] ==0) and (data["var31"]>=-1.33232117 and data["var31"] !=0) and (data["var57"]<-1.15386295 or data["var57"] ==0) and (data["var13"]>=-0.496492088 and data["var13"] !=0)):
             s.append("12144")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]<1.29275966 or data["var24"] ==0) and (data["var31"]>=-1.33232117 and data["var31"] !=0) and (data["var57"]>=-1.15386295 and data["var57"] !=0) and (data["var34"]<-1.87174416 or data["var34"] ==0)):
             s.append("12145")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]<1.29275966 or data["var24"] ==0) and (data["var31"]>=-1.33232117 and data["var31"] !=0) and (data["var57"]>=-1.15386295 and data["var57"] !=0) and (data["var34"]>=-1.87174416 and data["var34"] !=0)):
             s.append("12146")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]>=1.29275966 and data["var24"] !=0) and (data["var14"]<1.62356114 or data["var14"] ==0) and (data["var41"]<-0.988597393 or data["var41"] ==0)):
             s.append("1291")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]>=1.29275966 and data["var24"] !=0) and (data["var14"]<1.62356114 or data["var14"] ==0) and (data["var41"]>=-0.988597393 and data["var41"] !=0)):
             s.append("1292")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]>=1.29275966 and data["var24"] !=0) and (data["var14"]>=1.62356114 and data["var14"] !=0) and (data["var06"]<-0.287571132 or data["var06"] ==0)):
             s.append("1293")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]<1.07971239 or data["var52"] ==0) and (data["var49"]>=-0.495757073 and data["var49"] !=0) and (data["var24"]>=1.29275966 and data["var24"] !=0) and (data["var14"]>=1.62356114 and data["var14"] !=0) and (data["var06"]>=-0.287571132 and data["var06"] !=0)):
             s.append("1294")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]<-1.62681651 or data["var33"] ==0) and (data["var44"]<0.462493181 or data["var44"] ==0)):
             s.append("1227")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]<-1.62681651 or data["var33"] ==0) and (data["var44"]>=0.462493181 and data["var44"] !=0)):
             s.append("1228")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]>=-1.62681651 and data["var33"] !=0) and (data["var15"]<-0.962533772 or data["var15"] ==0) and (data["var60"]<-0.103683956 or data["var60"] ==0) and (data["var31"]<0.269503146 or data["var31"] ==0)):
             s.append("1295")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]>=-1.62681651 and data["var33"] !=0) and (data["var15"]<-0.962533772 or data["var15"] ==0) and (data["var60"]<-0.103683956 or data["var60"] ==0) and (data["var31"]>=0.269503146 and data["var31"] !=0)):
             s.append("1296")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]>=-1.62681651 and data["var33"] !=0) and (data["var15"]<-0.962533772 or data["var15"] ==0) and (data["var60"]>=-0.103683956 and data["var60"] !=0) and (data["var60"]<1.80100334 or data["var60"] ==0)):
             s.append("1297")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]>=-1.62681651 and data["var33"] !=0) and (data["var15"]<-0.962533772 or data["var15"] ==0) and (data["var60"]>=-0.103683956 and data["var60"] !=0) and (data["var60"]>=1.80100334 and data["var60"] !=0)):
             s.append("1298")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]>=-1.62681651 and data["var33"] !=0) and (data["var15"]>=-0.962533772 and data["var15"] !=0) and (data["var47"]<-1.62972045 or data["var47"] ==0)):
             s.append("1255")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]>=-1.62681651 and data["var33"] !=0) and (data["var15"]>=-0.962533772 and data["var15"] !=0) and (data["var47"]>=-1.62972045 and data["var47"] !=0) and (data["var19"]<2.31970596 or data["var19"] ==0)):
             s.append("1299")
    if((data["var39"]>=2.54962015 and data["var39"] !=0) and (data["var52"]>=1.07971239 and data["var52"] !=0) and (data["var33"]>=-1.62681651 and data["var33"] !=0) and (data["var15"]>=-0.962533772 and data["var15"] !=0) and (data["var47"]>=-1.62972045 and data["var47"] !=0) and (data["var19"]>=2.31970596 and data["var19"] !=0)):
             s.append("12100")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]<-0.0998236313 or data["var07"] ==0) and (data["var08"]<-0.985745132 or data["var08"] ==0) and (data["var06"]<1.37927008 or data["var06"] ==0)):
             s.append("1353")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]<-0.0998236313 or data["var07"] ==0) and (data["var08"]<-0.985745132 or data["var08"] ==0) and (data["var06"]>=1.37927008 and data["var06"] !=0)):
             s.append("1354")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]<-0.0998236313 or data["var07"] ==0) and (data["var08"]>=-0.985745132 and data["var08"] !=0) and (data["var32"]<-0.751752734 or data["var32"] ==0) and (data["var16"]<0.682458639 or data["var16"] ==0)):
             s.append("1387")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]<-0.0998236313 or data["var07"] ==0) and (data["var08"]>=-0.985745132 and data["var08"] !=0) and (data["var32"]<-0.751752734 or data["var32"] ==0) and (data["var16"]>=0.682458639 and data["var16"] !=0)):
             s.append("1388")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]<-0.0998236313 or data["var07"] ==0) and (data["var08"]>=-0.985745132 and data["var08"] !=0) and (data["var32"]>=-0.751752734 and data["var32"] !=0) and (data["var16"]<-0.112980992 or data["var16"] ==0)):
             s.append("1389")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]<-0.0998236313 or data["var07"] ==0) and (data["var08"]>=-0.985745132 and data["var08"] !=0) and (data["var32"]>=-0.751752734 and data["var32"] !=0) and (data["var16"]>=-0.112980992 and data["var16"] !=0)):
             s.append("1390")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]>=-0.0998236313 and data["var07"] !=0) and (data["var13"]<-1.73636854 or data["var13"] ==0)):
             s.append("1331")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]>=-0.0998236313 and data["var07"] !=0) and (data["var13"]>=-1.73636854 and data["var13"] !=0) and (data["var29"]<1.99485278 or data["var29"] ==0) and (data["var38"]<-1.27435946 or data["var38"] ==0)):
             s.append("1391")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]>=-0.0998236313 and data["var07"] !=0) and (data["var13"]>=-1.73636854 and data["var13"] !=0) and (data["var29"]<1.99485278 or data["var29"] ==0) and (data["var38"]>=-1.27435946 and data["var38"] !=0)):
             s.append("1392")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]<0.343625993 or data["var07"] ==0) and (data["var07"]>=-0.0998236313 and data["var07"] !=0) and (data["var13"]>=-1.73636854 and data["var13"] !=0) and (data["var29"]>=1.99485278 and data["var29"] !=0)):
             s.append("1358")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]<0.723049045 or data["var18"] ==0) and (data["var37"]<0.422551095 or data["var37"] ==0) and (data["var09"]<-0.880269289 or data["var09"] ==0) and (data["var36"]<1.81400967 or data["var36"] ==0)):
             s.append("1393")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]<0.723049045 or data["var18"] ==0) and (data["var37"]<0.422551095 or data["var37"] ==0) and (data["var09"]<-0.880269289 or data["var09"] ==0) and (data["var36"]>=1.81400967 and data["var36"] !=0)):
             s.append("1394")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]<0.723049045 or data["var18"] ==0) and (data["var37"]<0.422551095 or data["var37"] ==0) and (data["var09"]>=-0.880269289 and data["var09"] !=0) and (data["var33"]<2.17253113 or data["var33"] ==0)):
             s.append("1395")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]<0.723049045 or data["var18"] ==0) and (data["var37"]<0.422551095 or data["var37"] ==0) and (data["var09"]>=-0.880269289 and data["var09"] !=0) and (data["var33"]>=2.17253113 and data["var33"] !=0)):
             s.append("1396")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]<0.723049045 or data["var18"] ==0) and (data["var37"]>=0.422551095 and data["var37"] !=0) and (data["var35"]<1.12985671 or data["var35"] ==0) and (data["var50"]<-0.571820498 or data["var50"] ==0)):
             s.append("1397")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]<0.723049045 or data["var18"] ==0) and (data["var37"]>=0.422551095 and data["var37"] !=0) and (data["var35"]<1.12985671 or data["var35"] ==0) and (data["var50"]>=-0.571820498 and data["var50"] !=0)):
             s.append("1398")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]<0.723049045 or data["var18"] ==0) and (data["var37"]>=0.422551095 and data["var37"] !=0) and (data["var35"]>=1.12985671 and data["var35"] !=0)):
             s.append("1362")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]>=0.723049045 and data["var18"] !=0) and (data["var30"]<-1.91461229 or data["var30"] ==0)):
             s.append("1335")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]>=0.723049045 and data["var18"] !=0) and (data["var30"]>=-1.91461229 and data["var30"] !=0) and (data["var27"]<-2.06351876 or data["var27"] ==0)):
             s.append("1363")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]<-2.42053056 or data["var60"] ==0) and (data["var07"]>=0.343625993 and data["var07"] !=0) and (data["var18"]>=0.723049045 and data["var18"] !=0) and (data["var30"]>=-1.91461229 and data["var30"] !=0) and (data["var27"]>=-2.06351876 and data["var27"] !=0)):
             s.append("1364")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]<1.46397686 or data["var59"] ==0) and (data["var28"]<-2.41523266 or data["var28"] ==0) and (data["var19"]<-1.30335808 or data["var19"] ==0)):
             s.append("1365")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]<1.46397686 or data["var59"] ==0) and (data["var28"]<-2.41523266 or data["var28"] ==0) and (data["var19"]>=-1.30335808 and data["var19"] !=0)):
             s.append("1366")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]<1.46397686 or data["var59"] ==0) and (data["var28"]>=-2.41523266 and data["var28"] !=0) and (data["var42"]<-2.37591505 or data["var42"] ==0) and (data["var19"]<1.2412082 or data["var19"] ==0)):
             s.append("1399")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]<1.46397686 or data["var59"] ==0) and (data["var28"]>=-2.41523266 and data["var28"] !=0) and (data["var42"]<-2.37591505 or data["var42"] ==0) and (data["var19"]>=1.2412082 and data["var19"] !=0)):
             s.append("13100")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]<1.46397686 or data["var59"] ==0) and (data["var28"]>=-2.41523266 and data["var28"] !=0) and (data["var42"]>=-2.37591505 and data["var42"] !=0) and (data["var55"]<-2.56739593 or data["var55"] ==0)):
             s.append("13101")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]<1.46397686 or data["var59"] ==0) and (data["var28"]>=-2.41523266 and data["var28"] !=0) and (data["var42"]>=-2.37591505 and data["var42"] !=0) and (data["var55"]>=-2.56739593 and data["var55"] !=0)):
             s.append("13102")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]>=1.46397686 and data["var59"] !=0) and (data["var25"]<-0.184706599 or data["var25"] ==0) and (data["var41"]<-0.175608039 or data["var41"] ==0)):
             s.append("1369")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]>=1.46397686 and data["var59"] !=0) and (data["var25"]<-0.184706599 or data["var25"] ==0) and (data["var41"]>=-0.175608039 and data["var41"] !=0) and (data["var15"]<1.15655756 or data["var15"] ==0)):
             s.append("13103")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]>=1.46397686 and data["var59"] !=0) and (data["var25"]<-0.184706599 or data["var25"] ==0) and (data["var41"]>=-0.175608039 and data["var41"] !=0) and (data["var15"]>=1.15655756 and data["var15"] !=0)):
             s.append("13104")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]>=1.46397686 and data["var59"] !=0) and (data["var25"]>=-0.184706599 and data["var25"] !=0) and (data["var52"]<1.77251577 or data["var52"] ==0) and (data["var43"]<1.61730886 or data["var43"] ==0)):
             s.append("13105")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]>=1.46397686 and data["var59"] !=0) and (data["var25"]>=-0.184706599 and data["var25"] !=0) and (data["var52"]<1.77251577 or data["var52"] ==0) and (data["var43"]>=1.61730886 and data["var43"] !=0)):
             s.append("13106")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]<-2.0382309 or data["var02"] ==0) and (data["var59"]>=1.46397686 and data["var59"] !=0) and (data["var25"]>=-0.184706599 and data["var25"] !=0) and (data["var52"]>=1.77251577 and data["var52"] !=0)):
             s.append("1372")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]<-1.99035633 or data["var30"] ==0) and (data["var01"]<-1.89370942 or data["var01"] ==0) and (data["var43"]<1.41513753 or data["var43"] ==0)):
             s.append("13107")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]<-1.99035633 or data["var30"] ==0) and (data["var01"]<-1.89370942 or data["var01"] ==0) and (data["var43"]>=1.41513753 and data["var43"] !=0)):
             s.append("13108")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]<-1.99035633 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var53"]<2.02864552 or data["var53"] ==0)):
             s.append("13109")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]<-1.99035633 or data["var30"] ==0) and (data["var01"]>=-1.89370942 and data["var01"] !=0) and (data["var53"]>=2.02864552 and data["var53"] !=0)):
             s.append("13110")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]>=-1.99035633 and data["var30"] !=0) and (data["var30"]<1.52604389 or data["var30"] ==0) and (data["var47"]<1.78762889 or data["var47"] ==0)):
             s.append("13111")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]>=-1.99035633 and data["var30"] !=0) and (data["var30"]<1.52604389 or data["var30"] ==0) and (data["var47"]>=1.78762889 and data["var47"] !=0)):
             s.append("13112")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]>=-1.99035633 and data["var30"] !=0) and (data["var30"]>=1.52604389 and data["var30"] !=0) and (data["var57"]<-1.54960656 or data["var57"] ==0)):
             s.append("13113")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]<2.20607018 or data["var33"] ==0) and (data["var30"]>=-1.99035633 and data["var30"] !=0) and (data["var30"]>=1.52604389 and data["var30"] !=0) and (data["var57"]>=-1.54960656 and data["var57"] !=0)):
             s.append("13114")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]<-1.28486884 or data["var47"] ==0) and (data["var20"]<-0.104516745 or data["var20"] ==0) and (data["var14"]<-1.04793096 or data["var14"] ==0)):
             s.append("13115")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]<-1.28486884 or data["var47"] ==0) and (data["var20"]<-0.104516745 or data["var20"] ==0) and (data["var14"]>=-1.04793096 and data["var14"] !=0)):
             s.append("13116")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]<-1.28486884 or data["var47"] ==0) and (data["var20"]>=-0.104516745 and data["var20"] !=0) and (data["var12"]<-0.653889298 or data["var12"] ==0)):
             s.append("13117")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]<-1.28486884 or data["var47"] ==0) and (data["var20"]>=-0.104516745 and data["var20"] !=0) and (data["var12"]>=-0.653889298 and data["var12"] !=0)):
             s.append("13118")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]>=-1.28486884 and data["var47"] !=0) and (data["var23"]<-0.696337223 or data["var23"] ==0) and (data["var52"]<0.554766715 or data["var52"] ==0)):
             s.append("13119")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]>=-1.28486884 and data["var47"] !=0) and (data["var23"]<-0.696337223 or data["var23"] ==0) and (data["var52"]>=0.554766715 and data["var52"] !=0)):
             s.append("13120")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]>=-1.28486884 and data["var47"] !=0) and (data["var23"]>=-0.696337223 and data["var23"] !=0) and (data["var34"]<1.39784431 or data["var34"] ==0)):
             s.append("13121")
    if((data["var18"]<2.81039286 or data["var18"] ==0) and (data["var60"]>=-2.42053056 and data["var60"] !=0) and (data["var02"]>=-2.0382309 and data["var02"] !=0) and (data["var33"]>=2.20607018 and data["var33"] !=0) and (data["var47"]>=-1.28486884 and data["var47"] !=0) and (data["var23"]>=-0.696337223 and data["var23"] !=0) and (data["var34"]>=1.39784431 and data["var34"] !=0)):
             s.append("13122")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]<-0.308260858 or data["var23"] ==0) and (data["var03"]<0.572631478 or data["var03"] ==0) and (data["var12"]<-1.7003777 or data["var12"] ==0)):
             s.append("1345")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]<-0.308260858 or data["var23"] ==0) and (data["var03"]<0.572631478 or data["var03"] ==0) and (data["var12"]>=-1.7003777 and data["var12"] !=0) and (data["var48"]<0.761747718 or data["var48"] ==0) and (data["var57"]<-0.584404528 or data["var57"] ==0)):
             s.append("13123")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]<-0.308260858 or data["var23"] ==0) and (data["var03"]<0.572631478 or data["var03"] ==0) and (data["var12"]>=-1.7003777 and data["var12"] !=0) and (data["var48"]<0.761747718 or data["var48"] ==0) and (data["var57"]>=-0.584404528 and data["var57"] !=0)):
             s.append("13124")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]<-0.308260858 or data["var23"] ==0) and (data["var03"]<0.572631478 or data["var03"] ==0) and (data["var12"]>=-1.7003777 and data["var12"] !=0) and (data["var48"]>=0.761747718 and data["var48"] !=0) and (data["var18"]<3.30725789 or data["var18"] ==0)):
             s.append("13125")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]<-0.308260858 or data["var23"] ==0) and (data["var03"]<0.572631478 or data["var03"] ==0) and (data["var12"]>=-1.7003777 and data["var12"] !=0) and (data["var48"]>=0.761747718 and data["var48"] !=0) and (data["var18"]>=3.30725789 and data["var18"] !=0)):
             s.append("13126")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]<-0.308260858 or data["var23"] ==0) and (data["var03"]>=0.572631478 and data["var03"] !=0) and (data["var42"]<-0.932096183 or data["var42"] ==0)):
             s.append("1347")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]<-0.308260858 or data["var23"] ==0) and (data["var03"]>=0.572631478 and data["var03"] !=0) and (data["var42"]>=-0.932096183 and data["var42"] !=0)):
             s.append("1348")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]>=-0.308260858 and data["var23"] !=0) and (data["var56"]<0.542447448 or data["var56"] ==0) and (data["var03"]<1.98627925 or data["var03"] ==0) and (data["var41"]<-1.8089273 or data["var41"] ==0)):
             s.append("1383")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]>=-0.308260858 and data["var23"] !=0) and (data["var56"]<0.542447448 or data["var56"] ==0) and (data["var03"]<1.98627925 or data["var03"] ==0) and (data["var41"]>=-1.8089273 and data["var41"] !=0) and (data["var03"]<-2.76733375 or data["var03"] ==0)):
             s.append("13127")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]>=-0.308260858 and data["var23"] !=0) and (data["var56"]<0.542447448 or data["var56"] ==0) and (data["var03"]<1.98627925 or data["var03"] ==0) and (data["var41"]>=-1.8089273 and data["var41"] !=0) and (data["var03"]>=-2.76733375 and data["var03"] !=0)):
             s.append("13128")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]>=-0.308260858 and data["var23"] !=0) and (data["var56"]<0.542447448 or data["var56"] ==0) and (data["var03"]>=1.98627925 and data["var03"] !=0)):
             s.append("1350")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]>=-0.308260858 and data["var23"] !=0) and (data["var56"]>=0.542447448 and data["var56"] !=0) and (data["var33"]<0.533134758 or data["var33"] ==0) and (data["var52"]<-0.0775022134 or data["var52"] ==0)):
             s.append("1385")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]>=-0.308260858 and data["var23"] !=0) and (data["var56"]>=0.542447448 and data["var56"] !=0) and (data["var33"]<0.533134758 or data["var33"] ==0) and (data["var52"]>=-0.0775022134 and data["var52"] !=0)):
             s.append("1386")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]<0.840868235 or data["var30"] ==0) and (data["var23"]>=-0.308260858 and data["var23"] !=0) and (data["var56"]>=0.542447448 and data["var56"] !=0) and (data["var33"]>=0.533134758 and data["var33"] !=0)):
             s.append("1352")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]>=0.840868235 and data["var30"] !=0) and (data["var45"]<0.703831553 or data["var45"] ==0)):
             s.append("1313")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]>=0.840868235 and data["var30"] !=0) and (data["var45"]>=0.703831553 and data["var45"] !=0) and (data["var40"]<0.666025817 or data["var40"] ==0)):
             s.append("1327")
    if((data["var18"]>=2.81039286 and data["var18"] !=0) and (data["var30"]>=0.840868235 and data["var30"] !=0) and (data["var45"]>=0.703831553 and data["var45"] !=0) and (data["var40"]>=0.666025817 and data["var40"] !=0)):
             s.append("1328")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]<-0.888447404 or data["var56"] ==0) and (data["var41"]<1.44344449 or data["var41"] ==0) and (data["var43"]<-2.10401249 or data["var43"] ==0)):
             s.append("1415")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]<-0.888447404 or data["var56"] ==0) and (data["var41"]<1.44344449 or data["var41"] ==0) and (data["var43"]>=-2.10401249 and data["var43"] !=0)):
             s.append("1416")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]<-0.888447404 or data["var56"] ==0) and (data["var41"]>=1.44344449 and data["var41"] !=0)):
             s.append("148")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]<-1.86440539 or data["var19"] ==0)):
             s.append("149")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]<1.5316987 or data["var44"] ==0) and (data["var17"]<-1.30538988 or data["var17"] ==0) and (data["var15"]<0.926077485 or data["var15"] ==0)):
             s.append("1441")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]<1.5316987 or data["var44"] ==0) and (data["var17"]<-1.30538988 or data["var17"] ==0) and (data["var15"]>=0.926077485 and data["var15"] !=0)):
             s.append("1442")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]<1.5316987 or data["var44"] ==0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var23"]<-0.861607671 or data["var23"] ==0) and (data["var20"]<-1.30169868 or data["var20"] ==0)):
             s.append("1465")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]<1.5316987 or data["var44"] ==0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var23"]<-0.861607671 or data["var23"] ==0) and (data["var20"]>=-1.30169868 and data["var20"] !=0)):
             s.append("1466")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]<1.5316987 or data["var44"] ==0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var23"]>=-0.861607671 and data["var23"] !=0) and (data["var44"]<0.223945469 or data["var44"] ==0)):
             s.append("1467")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]<1.5316987 or data["var44"] ==0) and (data["var17"]>=-1.30538988 and data["var17"] !=0) and (data["var23"]>=-0.861607671 and data["var23"] !=0) and (data["var44"]>=0.223945469 and data["var44"] !=0)):
             s.append("1468")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]>=1.5316987 and data["var44"] !=0) and (data["var04"]<-1.28611469 or data["var04"] ==0)):
             s.append("1427")
    if((data["var14"]<-2.73300958 or data["var14"] ==0) and (data["var56"]>=-0.888447404 and data["var56"] !=0) and (data["var19"]>=-1.86440539 and data["var19"] !=0) and (data["var44"]>=1.5316987 and data["var44"] !=0) and (data["var04"]>=-1.28611469 and data["var04"] !=0)):
             s.append("1428")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]<-2.31924868 or data["var19"] ==0) and (data["var33"]<1.72231722 or data["var33"] ==0) and (data["var11"]<1.47547078 or data["var11"] ==0)):
             s.append("1469")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]<-2.31924868 or data["var19"] ==0) and (data["var33"]<1.72231722 or data["var33"] ==0) and (data["var11"]>=1.47547078 and data["var11"] !=0)):
             s.append("1470")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]<-2.31924868 or data["var19"] ==0) and (data["var33"]>=1.72231722 and data["var33"] !=0) and (data["var53"]<1.60896778 or data["var53"] ==0)):
             s.append("1471")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]<-2.31924868 or data["var19"] ==0) and (data["var33"]>=1.72231722 and data["var33"] !=0) and (data["var53"]>=1.60896778 and data["var53"] !=0)):
             s.append("1472")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]>=-2.31924868 and data["var19"] !=0) and (data["var19"]<1.20957637 or data["var19"] ==0) and (data["var44"]<-2.51208544 or data["var44"] ==0)):
             s.append("1473")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]>=-2.31924868 and data["var19"] !=0) and (data["var19"]<1.20957637 or data["var19"] ==0) and (data["var44"]>=-2.51208544 and data["var44"] !=0)):
             s.append("1474")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]>=-2.31924868 and data["var19"] !=0) and (data["var19"]>=1.20957637 and data["var19"] !=0) and (data["var41"]<1.52416265 or data["var41"] ==0)):
             s.append("1475")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]<1.77043474 or data["var55"] ==0) and (data["var19"]>=-2.31924868 and data["var19"] !=0) and (data["var19"]>=1.20957637 and data["var19"] !=0) and (data["var41"]>=1.52416265 and data["var41"] !=0)):
             s.append("1476")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]<-1.6273396 or data["var41"] ==0) and (data["var59"]<1.7266264 or data["var59"] ==0) and (data["var17"]<-2.39363861 or data["var17"] ==0)):
             s.append("1477")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]<-1.6273396 or data["var41"] ==0) and (data["var59"]<1.7266264 or data["var59"] ==0) and (data["var17"]>=-2.39363861 and data["var17"] !=0)):
             s.append("1478")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]<-1.6273396 or data["var41"] ==0) and (data["var59"]>=1.7266264 and data["var59"] !=0) and (data["var27"]<-0.238401204 or data["var27"] ==0)):
             s.append("1479")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]<-1.6273396 or data["var41"] ==0) and (data["var59"]>=1.7266264 and data["var59"] !=0) and (data["var27"]>=-0.238401204 and data["var27"] !=0)):
             s.append("1480")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]>=-1.6273396 and data["var41"] !=0) and (data["var55"]<3.54877615 or data["var55"] ==0) and (data["var03"]<0.634145498 or data["var03"] ==0)):
             s.append("1481")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]>=-1.6273396 and data["var41"] !=0) and (data["var55"]<3.54877615 or data["var55"] ==0) and (data["var03"]>=0.634145498 and data["var03"] !=0)):
             s.append("1482")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]>=-1.6273396 and data["var41"] !=0) and (data["var55"]>=3.54877615 and data["var55"] !=0) and (data["var43"]<1.0192399 or data["var43"] ==0)):
             s.append("1483")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]<2.43601036 or data["var58"] ==0) and (data["var55"]>=1.77043474 and data["var55"] !=0) and (data["var41"]>=-1.6273396 and data["var41"] !=0) and (data["var55"]>=3.54877615 and data["var55"] !=0) and (data["var43"]>=1.0192399 and data["var43"] !=0)):
             s.append("1484")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]<0.639394939 or data["var32"] ==0) and (data["var02"]<1.16081166 or data["var02"] ==0) and (data["var48"]<-0.612926245 or data["var48"] ==0)):
             s.append("1485")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]<0.639394939 or data["var32"] ==0) and (data["var02"]<1.16081166 or data["var02"] ==0) and (data["var48"]>=-0.612926245 and data["var48"] !=0)):
             s.append("1486")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]<0.639394939 or data["var32"] ==0) and (data["var02"]>=1.16081166 and data["var02"] !=0) and (data["var29"]<0.671318412 or data["var29"] ==0)):
             s.append("1487")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]<0.639394939 or data["var32"] ==0) and (data["var02"]>=1.16081166 and data["var02"] !=0) and (data["var29"]>=0.671318412 and data["var29"] !=0)):
             s.append("1488")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]>=0.639394939 and data["var32"] !=0) and (data["var02"]<-1.01396322 or data["var02"] ==0) and (data["var18"]<-0.36918205 or data["var18"] ==0)):
             s.append("1489")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]>=0.639394939 and data["var32"] !=0) and (data["var02"]<-1.01396322 or data["var02"] ==0) and (data["var18"]>=-0.36918205 and data["var18"] !=0)):
             s.append("1490")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]>=0.639394939 and data["var32"] !=0) and (data["var02"]>=-1.01396322 and data["var02"] !=0) and (data["var37"]<-1.63773608 or data["var37"] ==0)):
             s.append("1491")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]<0.587125778 or data["var15"] ==0) and (data["var32"]>=0.639394939 and data["var32"] !=0) and (data["var02"]>=-1.01396322 and data["var02"] !=0) and (data["var37"]>=-1.63773608 and data["var37"] !=0)):
             s.append("1492")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]>=0.587125778 and data["var15"] !=0) and (data["var20"]<1.87607205 or data["var20"] ==0) and (data["var32"]<-0.319172949 or data["var32"] ==0) and (data["var01"]<-1.83599031 or data["var01"] ==0)):
             s.append("1493")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]>=0.587125778 and data["var15"] !=0) and (data["var20"]<1.87607205 or data["var20"] ==0) and (data["var32"]<-0.319172949 or data["var32"] ==0) and (data["var01"]>=-1.83599031 and data["var01"] !=0)):
             s.append("1494")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]>=0.587125778 and data["var15"] !=0) and (data["var20"]<1.87607205 or data["var20"] ==0) and (data["var32"]>=-0.319172949 and data["var32"] !=0) and (data["var05"]<-0.406721354 or data["var05"] ==0)):
             s.append("1495")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]>=0.587125778 and data["var15"] !=0) and (data["var20"]<1.87607205 or data["var20"] ==0) and (data["var32"]>=-0.319172949 and data["var32"] !=0) and (data["var05"]>=-0.406721354 and data["var05"] !=0)):
             s.append("1496")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]<2.16538429 or data["var46"] ==0) and (data["var58"]>=2.43601036 and data["var58"] !=0) and (data["var15"]>=0.587125778 and data["var15"] !=0) and (data["var20"]>=1.87607205 and data["var20"] !=0)):
             s.append("1436")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]<-0.952416062 or data["var13"] ==0) and (data["var50"]<2.34298277 or data["var50"] ==0) and (data["var37"]<0.00980288349 or data["var37"] ==0) and (data["var53"]<-1.60692906 or data["var53"] ==0)):
             s.append("1497")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]<-0.952416062 or data["var13"] ==0) and (data["var50"]<2.34298277 or data["var50"] ==0) and (data["var37"]<0.00980288349 or data["var37"] ==0) and (data["var53"]>=-1.60692906 and data["var53"] !=0)):
             s.append("1498")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]<-0.952416062 or data["var13"] ==0) and (data["var50"]<2.34298277 or data["var50"] ==0) and (data["var37"]>=0.00980288349 and data["var37"] !=0) and (data["var55"]<2.02706242 or data["var55"] ==0)):
             s.append("1499")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]<-0.952416062 or data["var13"] ==0) and (data["var50"]<2.34298277 or data["var50"] ==0) and (data["var37"]>=0.00980288349 and data["var37"] !=0) and (data["var55"]>=2.02706242 and data["var55"] !=0)):
             s.append("14100")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]<-0.952416062 or data["var13"] ==0) and (data["var50"]>=2.34298277 and data["var50"] !=0)):
             s.append("1438")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]<1.24637365 or data["var34"] ==0) and (data["var17"]<1.91826332 or data["var17"] ==0) and (data["var28"]<1.1052916 or data["var28"] ==0)):
             s.append("14101")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]<1.24637365 or data["var34"] ==0) and (data["var17"]<1.91826332 or data["var17"] ==0) and (data["var28"]>=1.1052916 and data["var28"] !=0)):
             s.append("14102")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]<1.24637365 or data["var34"] ==0) and (data["var17"]>=1.91826332 and data["var17"] !=0) and (data["var53"]<1.37442911 or data["var53"] ==0)):
             s.append("14103")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]<1.24637365 or data["var34"] ==0) and (data["var17"]>=1.91826332 and data["var17"] !=0) and (data["var53"]>=1.37442911 and data["var53"] !=0)):
             s.append("14104")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]>=1.24637365 and data["var34"] !=0) and (data["var56"]<-1.83187532 or data["var56"] ==0) and (data["var25"]<-0.0305091031 or data["var25"] ==0)):
             s.append("14105")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]>=1.24637365 and data["var34"] !=0) and (data["var56"]<-1.83187532 or data["var56"] ==0) and (data["var25"]>=-0.0305091031 and data["var25"] !=0)):
             s.append("14106")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]>=1.24637365 and data["var34"] !=0) and (data["var56"]>=-1.83187532 and data["var56"] !=0) and (data["var55"]<0.91694963 or data["var55"] ==0)):
             s.append("14107")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]<2.41276312 or data["var37"] ==0) and (data["var13"]>=-0.952416062 and data["var13"] !=0) and (data["var34"]>=1.24637365 and data["var34"] !=0) and (data["var56"]>=-1.83187532 and data["var56"] !=0) and (data["var55"]>=0.91694963 and data["var55"] !=0)):
             s.append("14108")
    if((data["var14"]>=-2.73300958 and data["var14"] !=0) and (data["var46"]>=2.16538429 and data["var46"] !=0) and (data["var37"]>=2.41276312 and data["var37"] !=0)):
             s.append("1414")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]<-1.54509401 or data["var16"] ==0) and (data["var21"]<-2.18535185 or data["var21"] ==0)):
             s.append("157")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]<-1.54509401 or data["var16"] ==0) and (data["var21"]>=-2.18535185 and data["var21"] !=0) and (data["var56"]<-1.48441648 or data["var56"] ==0)):
             s.append("1515")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]<-1.54509401 or data["var16"] ==0) and (data["var21"]>=-2.18535185 and data["var21"] !=0) and (data["var56"]>=-1.48441648 and data["var56"] !=0)):
             s.append("1516")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]<-1.67610085 or data["var10"] ==0) and (data["var08"]<-0.791005075 or data["var08"] ==0)):
             s.append("1529")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]<-1.67610085 or data["var10"] ==0) and (data["var08"]>=-0.791005075 and data["var08"] !=0)):
             s.append("1530")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]<0.749709725 or data["var04"] ==0) and (data["var32"]<1.2045753 or data["var32"] ==0) and (data["var30"]<-1.07250476 or data["var30"] ==0)):
             s.append("1575")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]<0.749709725 or data["var04"] ==0) and (data["var32"]<1.2045753 or data["var32"] ==0) and (data["var30"]>=-1.07250476 and data["var30"] !=0)):
             s.append("1576")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]<0.749709725 or data["var04"] ==0) and (data["var32"]>=1.2045753 and data["var32"] !=0) and (data["var07"]<0.46423471 or data["var07"] ==0)):
             s.append("1577")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]<0.749709725 or data["var04"] ==0) and (data["var32"]>=1.2045753 and data["var32"] !=0) and (data["var07"]>=0.46423471 and data["var07"] !=0)):
             s.append("1578")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]>=0.749709725 and data["var04"] !=0) and (data["var09"]<-0.54963398 or data["var09"] ==0) and (data["var54"]<0.771584868 or data["var54"] ==0)):
             s.append("1579")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]>=0.749709725 and data["var04"] !=0) and (data["var09"]<-0.54963398 or data["var09"] ==0) and (data["var54"]>=0.771584868 and data["var54"] !=0)):
             s.append("1580")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]>=0.749709725 and data["var04"] !=0) and (data["var09"]>=-0.54963398 and data["var09"] !=0) and (data["var15"]<-1.56272149 or data["var15"] ==0)):
             s.append("1581")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]<1.21752739 or data["var48"] ==0) and (data["var10"]>=-1.67610085 and data["var10"] !=0) and (data["var04"]>=0.749709725 and data["var04"] !=0) and (data["var09"]>=-0.54963398 and data["var09"] !=0) and (data["var15"]>=-1.56272149 and data["var15"] !=0)):
             s.append("1582")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]>=1.21752739 and data["var48"] !=0) and (data["var11"]<1.13916743 or data["var11"] ==0) and (data["var24"]<-0.897018492 or data["var24"] ==0) and (data["var51"]<-3.2032361 or data["var51"] ==0)):
             s.append("1555")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]>=1.21752739 and data["var48"] !=0) and (data["var11"]<1.13916743 or data["var11"] ==0) and (data["var24"]<-0.897018492 or data["var24"] ==0) and (data["var51"]>=-3.2032361 and data["var51"] !=0)):
             s.append("1556")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]>=1.21752739 and data["var48"] !=0) and (data["var11"]<1.13916743 or data["var11"] ==0) and (data["var24"]>=-0.897018492 and data["var24"] !=0)):
             s.append("1534")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]>=1.21752739 and data["var48"] !=0) and (data["var11"]>=1.13916743 and data["var11"] !=0) and (data["var02"]<0.148869425 or data["var02"] ==0)):
             s.append("1535")
    if((data["var51"]<-2.5842433 or data["var51"] ==0) and (data["var16"]>=-1.54509401 and data["var16"] !=0) and (data["var48"]>=1.21752739 and data["var48"] !=0) and (data["var11"]>=1.13916743 and data["var11"] !=0) and (data["var02"]>=0.148869425 and data["var02"] !=0)):
             s.append("1536")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]<-1.14116812 or data["var46"] ==0) and (data["var09"]<-1.5726248 or data["var09"] ==0)):
             s.append("1521")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]<-1.14116812 or data["var46"] ==0) and (data["var09"]>=-1.5726248 and data["var09"] !=0) and (data["var50"]<-1.3012464 or data["var50"] ==0)):
             s.append("1537")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]<-1.14116812 or data["var46"] ==0) and (data["var09"]>=-1.5726248 and data["var09"] !=0) and (data["var50"]>=-1.3012464 and data["var50"] !=0)):
             s.append("1538")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]<1.74528909 or data["var44"] ==0) and (data["var39"]<-1.03486776 or data["var39"] ==0) and (data["var16"]<1.59264672 or data["var16"] ==0) and (data["var53"]<1.16150725 or data["var53"] ==0)):
             s.append("1583")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]<1.74528909 or data["var44"] ==0) and (data["var39"]<-1.03486776 or data["var39"] ==0) and (data["var16"]<1.59264672 or data["var16"] ==0) and (data["var53"]>=1.16150725 and data["var53"] !=0)):
             s.append("1584")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]<1.74528909 or data["var44"] ==0) and (data["var39"]<-1.03486776 or data["var39"] ==0) and (data["var16"]>=1.59264672 and data["var16"] !=0)):
             s.append("1558")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]<1.74528909 or data["var44"] ==0) and (data["var39"]>=-1.03486776 and data["var39"] !=0) and (data["var39"]<0.881893396 or data["var39"] ==0) and (data["var51"]<-1.27332294 or data["var51"] ==0)):
             s.append("1585")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]<1.74528909 or data["var44"] ==0) and (data["var39"]>=-1.03486776 and data["var39"] !=0) and (data["var39"]<0.881893396 or data["var39"] ==0) and (data["var51"]>=-1.27332294 and data["var51"] !=0)):
             s.append("1586")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]<1.74528909 or data["var44"] ==0) and (data["var39"]>=-1.03486776 and data["var39"] !=0) and (data["var39"]>=0.881893396 and data["var39"] !=0) and (data["var27"]<1.15340877 or data["var27"] ==0)):
             s.append("1587")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]<1.74528909 or data["var44"] ==0) and (data["var39"]>=-1.03486776 and data["var39"] !=0) and (data["var39"]>=0.881893396 and data["var39"] !=0) and (data["var27"]>=1.15340877 and data["var27"] !=0)):
             s.append("1588")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]>=1.74528909 and data["var44"] !=0) and (data["var08"]<1.57151866 or data["var08"] ==0) and (data["var10"]<-1.0362016 or data["var10"] ==0)):
             s.append("1561")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]>=1.74528909 and data["var44"] !=0) and (data["var08"]<1.57151866 or data["var08"] ==0) and (data["var10"]>=-1.0362016 and data["var10"] !=0)):
             s.append("1562")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]<-2.58883095 or data["var48"] ==0) and (data["var46"]>=-1.14116812 and data["var46"] !=0) and (data["var44"]>=1.74528909 and data["var44"] !=0) and (data["var08"]>=1.57151866 and data["var08"] !=0)):
             s.append("1542")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]<1.40252566 or data["var15"] ==0) and (data["var33"]<-1.03072453 or data["var33"] ==0) and (data["var31"]<-1.61902571 or data["var31"] ==0)):
             s.append("1589")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]<1.40252566 or data["var15"] ==0) and (data["var33"]<-1.03072453 or data["var33"] ==0) and (data["var31"]>=-1.61902571 and data["var31"] !=0)):
             s.append("1590")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]<1.40252566 or data["var15"] ==0) and (data["var33"]>=-1.03072453 and data["var33"] !=0) and (data["var53"]<1.09222758 or data["var53"] ==0)):
             s.append("1591")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]<1.40252566 or data["var15"] ==0) and (data["var33"]>=-1.03072453 and data["var33"] !=0) and (data["var53"]>=1.09222758 and data["var53"] !=0)):
             s.append("1592")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]>=1.40252566 and data["var15"] !=0) and (data["var33"]<-1.75775576 or data["var33"] ==0) and (data["var56"]<0.116990536 or data["var56"] ==0)):
             s.append("1593")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]>=1.40252566 and data["var15"] !=0) and (data["var33"]<-1.75775576 or data["var33"] ==0) and (data["var56"]>=0.116990536 and data["var56"] !=0)):
             s.append("1594")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]>=1.40252566 and data["var15"] !=0) and (data["var33"]>=-1.75775576 and data["var33"] !=0) and (data["var08"]<1.54640317 or data["var08"] ==0)):
             s.append("1595")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]<1.78389192 or data["var40"] ==0) and (data["var15"]>=1.40252566 and data["var15"] !=0) and (data["var33"]>=-1.75775576 and data["var33"] !=0) and (data["var08"]>=1.54640317 and data["var08"] !=0)):
             s.append("1596")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]>=1.78389192 and data["var40"] !=0) and (data["var42"]<-1.23757434 or data["var42"] ==0)):
             s.append("1545")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]<-2.33265209 or data["var07"] ==0) and (data["var40"]>=1.78389192 and data["var40"] !=0) and (data["var42"]>=-1.23757434 and data["var42"] !=0)):
             s.append("1546")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]<-2.30706 or data["var12"] ==0) and (data["var12"]<-3.11337233 or data["var12"] ==0) and (data["var15"]<-0.547372341 or data["var15"] ==0)):
             s.append("1597")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]<-2.30706 or data["var12"] ==0) and (data["var12"]<-3.11337233 or data["var12"] ==0) and (data["var15"]>=-0.547372341 and data["var15"] !=0)):
             s.append("1598")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]<-2.30706 or data["var12"] ==0) and (data["var12"]>=-3.11337233 and data["var12"] !=0) and (data["var31"]<-1.56685042 or data["var31"] ==0)):
             s.append("1599")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]<-2.30706 or data["var12"] ==0) and (data["var12"]>=-3.11337233 and data["var12"] !=0) and (data["var31"]>=-1.56685042 and data["var31"] !=0)):
             s.append("15100")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]>=-2.30706 and data["var12"] !=0) and (data["var47"]<-2.63812542 or data["var47"] ==0) and (data["var20"]<1.29715586 or data["var20"] ==0)):
             s.append("15101")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]>=-2.30706 and data["var12"] !=0) and (data["var47"]<-2.63812542 or data["var47"] ==0) and (data["var20"]>=1.29715586 and data["var20"] !=0)):
             s.append("15102")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]>=-2.30706 and data["var12"] !=0) and (data["var47"]>=-2.63812542 and data["var47"] !=0) and (data["var14"]<-2.73768997 or data["var14"] ==0)):
             s.append("15103")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]<2.81632471 or data["var38"] ==0) and (data["var12"]>=-2.30706 and data["var12"] !=0) and (data["var47"]>=-2.63812542 and data["var47"] !=0) and (data["var14"]>=-2.73768997 and data["var14"] !=0)):
             s.append("15104")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]>=2.81632471 and data["var38"] !=0) and (data["var29"]<-0.832231522 or data["var29"] ==0) and (data["var54"]<-1.93983734 or data["var54"] ==0)):
             s.append("1571")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]>=2.81632471 and data["var38"] !=0) and (data["var29"]<-0.832231522 or data["var29"] ==0) and (data["var54"]>=-1.93983734 and data["var54"] !=0) and (data["var10"]<1.37548447 or data["var10"] ==0)):
             s.append("15105")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]>=2.81632471 and data["var38"] !=0) and (data["var29"]<-0.832231522 or data["var29"] ==0) and (data["var54"]>=-1.93983734 and data["var54"] !=0) and (data["var10"]>=1.37548447 and data["var10"] !=0)):
             s.append("15106")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]>=2.81632471 and data["var38"] !=0) and (data["var29"]>=-0.832231522 and data["var29"] !=0) and (data["var53"]<-0.510335922 or data["var53"] ==0) and (data["var08"]<0.549559712 or data["var08"] ==0)):
             s.append("15107")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]>=2.81632471 and data["var38"] !=0) and (data["var29"]>=-0.832231522 and data["var29"] !=0) and (data["var53"]<-0.510335922 or data["var53"] ==0) and (data["var08"]>=0.549559712 and data["var08"] !=0)):
             s.append("15108")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]>=2.81632471 and data["var38"] !=0) and (data["var29"]>=-0.832231522 and data["var29"] !=0) and (data["var53"]>=-0.510335922 and data["var53"] !=0) and (data["var29"]<-0.324286163 or data["var29"] ==0)):
             s.append("15109")
    if((data["var51"]>=-2.5842433 and data["var51"] !=0) and (data["var48"]>=-2.58883095 and data["var48"] !=0) and (data["var07"]>=-2.33265209 and data["var07"] !=0) and (data["var38"]>=2.81632471 and data["var38"] !=0) and (data["var29"]>=-0.832231522 and data["var29"] !=0) and (data["var53"]>=-0.510335922 and data["var53"] !=0) and (data["var29"]>=-0.324286163 and data["var29"] !=0)):
             s.append("15110")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]<-1.00621939 or data["var02"] ==0) and (data["var11"]<-1.77234292 or data["var11"] ==0) and (data["var22"]<0.242594898 or data["var22"] ==0)):
             s.append("1615")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]<-1.00621939 or data["var02"] ==0) and (data["var11"]<-1.77234292 or data["var11"] ==0) and (data["var22"]>=0.242594898 and data["var22"] !=0)):
             s.append("1616")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]<-1.00621939 or data["var02"] ==0) and (data["var11"]>=-1.77234292 and data["var11"] !=0) and (data["var44"]<-1.06693435 or data["var44"] ==0) and (data["var55"]<-0.192922682 or data["var55"] ==0)):
             s.append("1631")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]<-1.00621939 or data["var02"] ==0) and (data["var11"]>=-1.77234292 and data["var11"] !=0) and (data["var44"]<-1.06693435 or data["var44"] ==0) and (data["var55"]>=-0.192922682 and data["var55"] !=0)):
             s.append("1632")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]<-1.00621939 or data["var02"] ==0) and (data["var11"]>=-1.77234292 and data["var11"] !=0) and (data["var44"]>=-1.06693435 and data["var44"] !=0) and (data["var50"]<-0.970615149 or data["var50"] ==0) and (data["var52"]<0.12708725 or data["var52"] ==0)):
             s.append("1655")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]<-1.00621939 or data["var02"] ==0) and (data["var11"]>=-1.77234292 and data["var11"] !=0) and (data["var44"]>=-1.06693435 and data["var44"] !=0) and (data["var50"]<-0.970615149 or data["var50"] ==0) and (data["var52"]>=0.12708725 and data["var52"] !=0)):
             s.append("1656")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]<-1.00621939 or data["var02"] ==0) and (data["var11"]>=-1.77234292 and data["var11"] !=0) and (data["var44"]>=-1.06693435 and data["var44"] !=0) and (data["var50"]>=-0.970615149 and data["var50"] !=0)):
             s.append("1634")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]<-1.52260399 or data["var16"] ==0) and (data["var15"]<1.43783152 or data["var15"] ==0) and (data["var37"]<1.67781878 or data["var37"] ==0)):
             s.append("1635")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]<-1.52260399 or data["var16"] ==0) and (data["var15"]<1.43783152 or data["var15"] ==0) and (data["var37"]>=1.67781878 and data["var37"] !=0)):
             s.append("1636")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]<-1.52260399 or data["var16"] ==0) and (data["var15"]>=1.43783152 and data["var15"] !=0)):
             s.append("1620")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]<1.35236728 or data["var25"] ==0) and (data["var58"]<-1.93227744 or data["var58"] ==0)):
             s.append("1637")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]<1.35236728 or data["var25"] ==0) and (data["var58"]>=-1.93227744 and data["var58"] !=0) and (data["var34"]<-1.20766807 or data["var34"] ==0) and (data["var22"]<-0.0324647278 or data["var22"] ==0)):
             s.append("1685")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]<1.35236728 or data["var25"] ==0) and (data["var58"]>=-1.93227744 and data["var58"] !=0) and (data["var34"]<-1.20766807 or data["var34"] ==0) and (data["var22"]>=-0.0324647278 and data["var22"] !=0)):
             s.append("1686")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]<1.35236728 or data["var25"] ==0) and (data["var58"]>=-1.93227744 and data["var58"] !=0) and (data["var34"]>=-1.20766807 and data["var34"] !=0) and (data["var30"]<1.4211328 or data["var30"] ==0)):
             s.append("1687")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]<1.35236728 or data["var25"] ==0) and (data["var58"]>=-1.93227744 and data["var58"] !=0) and (data["var34"]>=-1.20766807 and data["var34"] !=0) and (data["var30"]>=1.4211328 and data["var30"] !=0)):
             s.append("1688")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]>=1.35236728 and data["var25"] !=0) and (data["var15"]<1.01966727 or data["var15"] ==0) and (data["var10"]<1.29424942 or data["var10"] ==0)):
             s.append("1659")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]>=1.35236728 and data["var25"] !=0) and (data["var15"]<1.01966727 or data["var15"] ==0) and (data["var10"]>=1.29424942 and data["var10"] !=0)):
             s.append("1660")
    if((data["var28"]<-2.52795053 or data["var28"] ==0) and (data["var02"]>=-1.00621939 and data["var02"] !=0) and (data["var16"]>=-1.52260399 and data["var16"] !=0) and (data["var25"]>=1.35236728 and data["var25"] !=0) and (data["var15"]>=1.01966727 and data["var15"] !=0)):
             s.append("1640")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]<-0.767213345 or data["var14"] ==0) and (data["var25"]<1.18380237 or data["var25"] ==0) and (data["var51"]<-1.92199159 or data["var51"] ==0) and (data["var53"]<0.227410361 or data["var53"] ==0)):
             s.append("1689")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]<-0.767213345 or data["var14"] ==0) and (data["var25"]<1.18380237 or data["var25"] ==0) and (data["var51"]<-1.92199159 or data["var51"] ==0) and (data["var53"]>=0.227410361 and data["var53"] !=0)):
             s.append("1690")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]<-0.767213345 or data["var14"] ==0) and (data["var25"]<1.18380237 or data["var25"] ==0) and (data["var51"]>=-1.92199159 and data["var51"] !=0) and (data["var49"]<-1.08765483 or data["var49"] ==0)):
             s.append("1691")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]<-0.767213345 or data["var14"] ==0) and (data["var25"]<1.18380237 or data["var25"] ==0) and (data["var51"]>=-1.92199159 and data["var51"] !=0) and (data["var49"]>=-1.08765483 and data["var49"] !=0)):
             s.append("1692")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]<-0.767213345 or data["var14"] ==0) and (data["var25"]>=1.18380237 and data["var25"] !=0) and (data["var22"]<0.242386252 or data["var22"] ==0)):
             s.append("1663")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]<-0.767213345 or data["var14"] ==0) and (data["var25"]>=1.18380237 and data["var25"] !=0) and (data["var22"]>=0.242386252 and data["var22"] !=0)):
             s.append("1664")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]<-1.01835442 or data["var59"] ==0) and (data["var30"]<0.881162047 or data["var30"] ==0) and (data["var12"]<1.09543872 or data["var12"] ==0)):
             s.append("1693")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]<-1.01835442 or data["var59"] ==0) and (data["var30"]<0.881162047 or data["var30"] ==0) and (data["var12"]>=1.09543872 and data["var12"] !=0)):
             s.append("1694")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]<-1.01835442 or data["var59"] ==0) and (data["var30"]>=0.881162047 and data["var30"] !=0) and (data["var48"]<-0.500961721 or data["var48"] ==0)):
             s.append("1695")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]<-1.01835442 or data["var59"] ==0) and (data["var30"]>=0.881162047 and data["var30"] !=0) and (data["var48"]>=-0.500961721 and data["var48"] !=0)):
             s.append("1696")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]>=-1.01835442 and data["var59"] !=0) and (data["var31"]<-1.19483519 or data["var31"] ==0) and (data["var20"]<-0.653993189 or data["var20"] ==0)):
             s.append("1697")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]>=-1.01835442 and data["var59"] !=0) and (data["var31"]<-1.19483519 or data["var31"] ==0) and (data["var20"]>=-0.653993189 and data["var20"] !=0)):
             s.append("1698")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]>=-1.01835442 and data["var59"] !=0) and (data["var31"]>=-1.19483519 and data["var31"] !=0) and (data["var60"]<-2.06322169 or data["var60"] ==0)):
             s.append("1699")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]<0.632641315 or data["var55"] ==0) and (data["var14"]>=-0.767213345 and data["var14"] !=0) and (data["var59"]>=-1.01835442 and data["var59"] !=0) and (data["var31"]>=-1.19483519 and data["var31"] !=0) and (data["var60"]>=-2.06322169 and data["var60"] !=0)):
             s.append("16100")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]<-2.07737732 or data["var28"] ==0)):
             s.append("1625")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]>=-2.07737732 and data["var28"] !=0) and (data["var53"]<0.443526685 or data["var53"] ==0) and (data["var49"]<-0.816936493 or data["var49"] ==0) and (data["var24"]<-0.237466305 or data["var24"] ==0)):
             s.append("16101")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]>=-2.07737732 and data["var28"] !=0) and (data["var53"]<0.443526685 or data["var53"] ==0) and (data["var49"]<-0.816936493 or data["var49"] ==0) and (data["var24"]>=-0.237466305 and data["var24"] !=0)):
             s.append("16102")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]>=-2.07737732 and data["var28"] !=0) and (data["var53"]<0.443526685 or data["var53"] ==0) and (data["var49"]>=-0.816936493 and data["var49"] !=0)):
             s.append("1670")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]>=-2.07737732 and data["var28"] !=0) and (data["var53"]>=0.443526685 and data["var53"] !=0) and (data["var57"]<0.323671818 or data["var57"] ==0) and (data["var32"]<0.666902006 or data["var32"] ==0)):
             s.append("16103")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]>=-2.07737732 and data["var28"] !=0) and (data["var53"]>=0.443526685 and data["var53"] !=0) and (data["var57"]<0.323671818 or data["var57"] ==0) and (data["var32"]>=0.666902006 and data["var32"] !=0)):
             s.append("16104")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]>=-2.07737732 and data["var28"] !=0) and (data["var53"]>=0.443526685 and data["var53"] !=0) and (data["var57"]>=0.323671818 and data["var57"] !=0) and (data["var06"]<0.379819632 or data["var06"] ==0)):
             s.append("16105")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]<-2.44961905 or data["var38"] ==0) and (data["var55"]>=0.632641315 and data["var55"] !=0) and (data["var28"]>=-2.07737732 and data["var28"] !=0) and (data["var53"]>=0.443526685 and data["var53"] !=0) and (data["var57"]>=0.323671818 and data["var57"] !=0) and (data["var06"]>=0.379819632 and data["var06"] !=0)):
             s.append("16106")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]<-1.8215816 or data["var34"] ==0) and (data["var06"]<1.19440508 or data["var06"] ==0) and (data["var02"]<-2.17197371 or data["var02"] ==0) and (data["var48"]<1.46258259 or data["var48"] ==0)):
             s.append("16107")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]<-1.8215816 or data["var34"] ==0) and (data["var06"]<1.19440508 or data["var06"] ==0) and (data["var02"]<-2.17197371 or data["var02"] ==0) and (data["var48"]>=1.46258259 and data["var48"] !=0)):
             s.append("16108")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]<-1.8215816 or data["var34"] ==0) and (data["var06"]<1.19440508 or data["var06"] ==0) and (data["var02"]>=-2.17197371 and data["var02"] !=0) and (data["var07"]<-2.0596838 or data["var07"] ==0)):
             s.append("16109")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]<-1.8215816 or data["var34"] ==0) and (data["var06"]<1.19440508 or data["var06"] ==0) and (data["var02"]>=-2.17197371 and data["var02"] !=0) and (data["var07"]>=-2.0596838 and data["var07"] !=0)):
             s.append("16110")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]<-1.8215816 or data["var34"] ==0) and (data["var06"]>=1.19440508 and data["var06"] !=0) and (data["var57"]<1.28177035 or data["var57"] ==0) and (data["var51"]<0.575363755 or data["var51"] ==0)):
             s.append("16111")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]<-1.8215816 or data["var34"] ==0) and (data["var06"]>=1.19440508 and data["var06"] !=0) and (data["var57"]<1.28177035 or data["var57"] ==0) and (data["var51"]>=0.575363755 and data["var51"] !=0)):
             s.append("16112")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]<-1.8215816 or data["var34"] ==0) and (data["var06"]>=1.19440508 and data["var06"] !=0) and (data["var57"]>=1.28177035 and data["var57"] !=0)):
             s.append("1676")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]<-2.87047338 or data["var41"] ==0) and (data["var54"]<-1.1497649 or data["var54"] ==0) and (data["var13"]<-0.249554411 or data["var13"] ==0)):
             s.append("16113")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]<-2.87047338 or data["var41"] ==0) and (data["var54"]<-1.1497649 or data["var54"] ==0) and (data["var13"]>=-0.249554411 and data["var13"] !=0)):
             s.append("16114")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]<-2.87047338 or data["var41"] ==0) and (data["var54"]>=-1.1497649 and data["var54"] !=0) and (data["var60"]<2.24682331 or data["var60"] ==0)):
             s.append("16115")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]<-2.87047338 or data["var41"] ==0) and (data["var54"]>=-1.1497649 and data["var54"] !=0) and (data["var60"]>=2.24682331 and data["var60"] !=0)):
             s.append("16116")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]>=-2.87047338 and data["var41"] !=0) and (data["var44"]<-2.7849679 or data["var44"] ==0) and (data["var53"]<1.25470018 or data["var53"] ==0)):
             s.append("16117")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]>=-2.87047338 and data["var41"] !=0) and (data["var44"]<-2.7849679 or data["var44"] ==0) and (data["var53"]>=1.25470018 and data["var53"] !=0)):
             s.append("16118")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]>=-2.87047338 and data["var41"] !=0) and (data["var44"]>=-2.7849679 and data["var44"] !=0) and (data["var37"]<2.41016912 or data["var37"] ==0)):
             s.append("16119")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]<2.22799397 or data["var32"] ==0) and (data["var34"]>=-1.8215816 and data["var34"] !=0) and (data["var41"]>=-2.87047338 and data["var41"] !=0) and (data["var44"]>=-2.7849679 and data["var44"] !=0) and (data["var37"]>=2.41016912 and data["var37"] !=0)):
             s.append("16120")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]<-0.79146862 or data["var36"] ==0) and (data["var02"]<2.15501928 or data["var02"] ==0) and (data["var39"]<-2.87910295 or data["var39"] ==0)):
             s.append("1681")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]<-0.79146862 or data["var36"] ==0) and (data["var02"]<2.15501928 or data["var02"] ==0) and (data["var39"]>=-2.87910295 and data["var39"] !=0) and (data["var22"]<-2.53491783 or data["var22"] ==0)):
             s.append("16121")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]<-0.79146862 or data["var36"] ==0) and (data["var02"]<2.15501928 or data["var02"] ==0) and (data["var39"]>=-2.87910295 and data["var39"] !=0) and (data["var22"]>=-2.53491783 and data["var22"] !=0)):
             s.append("16122")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]<-0.79146862 or data["var36"] ==0) and (data["var02"]>=2.15501928 and data["var02"] !=0)):
             s.append("1652")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]>=-0.79146862 and data["var36"] !=0) and (data["var22"]<2.10723543 or data["var22"] ==0) and (data["var32"]<3.32086468 or data["var32"] ==0) and (data["var34"]<-0.210921019 or data["var34"] ==0)):
             s.append("16123")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]>=-0.79146862 and data["var36"] !=0) and (data["var22"]<2.10723543 or data["var22"] ==0) and (data["var32"]<3.32086468 or data["var32"] ==0) and (data["var34"]>=-0.210921019 and data["var34"] !=0)):
             s.append("16124")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]>=-0.79146862 and data["var36"] !=0) and (data["var22"]<2.10723543 or data["var22"] ==0) and (data["var32"]>=3.32086468 and data["var32"] !=0) and (data["var22"]<-0.381564498 or data["var22"] ==0)):
             s.append("16125")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]>=-0.79146862 and data["var36"] !=0) and (data["var22"]<2.10723543 or data["var22"] ==0) and (data["var32"]>=3.32086468 and data["var32"] !=0) and (data["var22"]>=-0.381564498 and data["var22"] !=0)):
             s.append("16126")
    if((data["var28"]>=-2.52795053 and data["var28"] !=0) and (data["var38"]>=-2.44961905 and data["var38"] !=0) and (data["var32"]>=2.22799397 and data["var32"] !=0) and (data["var36"]>=-0.79146862 and data["var36"] !=0) and (data["var22"]>=2.10723543 and data["var22"] !=0)):
             s.append("1654")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]<-2.44539833 or data["var51"] ==0) and (data["var38"]<-1.50689292 or data["var38"] ==0)):
             s.append("1729")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]<-2.44539833 or data["var51"] ==0) and (data["var38"]>=-1.50689292 and data["var38"] !=0)):
             s.append("1730")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]>=-2.44539833 and data["var51"] !=0) and (data["var43"]<1.82927692 or data["var43"] ==0) and (data["var24"]<1.65119219 or data["var24"] ==0) and (data["var22"]<1.80544567 or data["var22"] ==0)):
             s.append("1789")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]>=-2.44539833 and data["var51"] !=0) and (data["var43"]<1.82927692 or data["var43"] ==0) and (data["var24"]<1.65119219 or data["var24"] ==0) and (data["var22"]>=1.80544567 and data["var22"] !=0)):
             s.append("1790")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]>=-2.44539833 and data["var51"] !=0) and (data["var43"]<1.82927692 or data["var43"] ==0) and (data["var24"]>=1.65119219 and data["var24"] !=0) and (data["var44"]<1.36147308 or data["var44"] ==0)):
             s.append("1791")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]>=-2.44539833 and data["var51"] !=0) and (data["var43"]<1.82927692 or data["var43"] ==0) and (data["var24"]>=1.65119219 and data["var24"] !=0) and (data["var44"]>=1.36147308 and data["var44"] !=0)):
             s.append("1792")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]>=-2.44539833 and data["var51"] !=0) and (data["var43"]>=1.82927692 and data["var43"] !=0) and (data["var46"]<0.092295073 or data["var46"] ==0)):
             s.append("1757")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]>=-2.44539833 and data["var51"] !=0) and (data["var43"]>=1.82927692 and data["var43"] !=0) and (data["var46"]>=0.092295073 and data["var46"] !=0) and (data["var37"]<0.47925368 or data["var37"] ==0)):
             s.append("1793")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]<0.975949645 or data["var18"] ==0) and (data["var51"]>=-2.44539833 and data["var51"] !=0) and (data["var43"]>=1.82927692 and data["var43"] !=0) and (data["var46"]>=0.092295073 and data["var46"] !=0) and (data["var37"]>=0.47925368 and data["var37"] !=0)):
             s.append("1794")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]<-0.650976241 or data["var45"] ==0) and (data["var54"]<1.83674693 or data["var54"] ==0) and (data["var27"]<-1.51785684 or data["var27"] ==0) and (data["var08"]<0.186869353 or data["var08"] ==0)):
             s.append("1795")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]<-0.650976241 or data["var45"] ==0) and (data["var54"]<1.83674693 or data["var54"] ==0) and (data["var27"]<-1.51785684 or data["var27"] ==0) and (data["var08"]>=0.186869353 and data["var08"] !=0)):
             s.append("1796")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]<-0.650976241 or data["var45"] ==0) and (data["var54"]<1.83674693 or data["var54"] ==0) and (data["var27"]>=-1.51785684 and data["var27"] !=0)):
             s.append("1760")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]<-0.650976241 or data["var45"] ==0) and (data["var54"]>=1.83674693 and data["var54"] !=0)):
             s.append("1734")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]<-0.681335449 or data["var52"] ==0) and (data["var58"]<-0.893297136 or data["var58"] ==0) and (data["var38"]<0.238080904 or data["var38"] ==0)):
             s.append("1797")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]<-0.681335449 or data["var52"] ==0) and (data["var58"]<-0.893297136 or data["var58"] ==0) and (data["var38"]>=0.238080904 and data["var38"] !=0)):
             s.append("1798")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]<-0.681335449 or data["var52"] ==0) and (data["var58"]>=-0.893297136 and data["var58"] !=0) and (data["var03"]<-2.09489107 or data["var03"] ==0)):
             s.append("1799")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]<-0.681335449 or data["var52"] ==0) and (data["var58"]>=-0.893297136 and data["var58"] !=0) and (data["var03"]>=-2.09489107 and data["var03"] !=0)):
             s.append("17100")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]>=-0.681335449 and data["var52"] !=0) and (data["var15"]<1.30635452 or data["var15"] ==0) and (data["var56"]<0.197325975 or data["var56"] ==0)):
             s.append("17101")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]>=-0.681335449 and data["var52"] !=0) and (data["var15"]<1.30635452 or data["var15"] ==0) and (data["var56"]>=0.197325975 and data["var56"] !=0)):
             s.append("17102")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]>=-0.681335449 and data["var52"] !=0) and (data["var15"]>=1.30635452 and data["var15"] !=0) and (data["var33"]<1.50712025 or data["var33"] ==0)):
             s.append("17103")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]<-1.92100489 or data["var16"] ==0) and (data["var18"]>=0.975949645 and data["var18"] !=0) and (data["var45"]>=-0.650976241 and data["var45"] !=0) and (data["var52"]>=-0.681335449 and data["var52"] !=0) and (data["var15"]>=1.30635452 and data["var15"] !=0) and (data["var33"]>=1.50712025 and data["var33"] !=0)):
             s.append("17104")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]<-3.56898403 or data["var43"] ==0) and (data["var03"]<1.29149079 or data["var03"] ==0) and (data["var44"]<1.87425852 or data["var44"] ==0)):
             s.append("1765")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]<-3.56898403 or data["var43"] ==0) and (data["var03"]<1.29149079 or data["var03"] ==0) and (data["var44"]>=1.87425852 and data["var44"] !=0)):
             s.append("1766")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]<-3.56898403 or data["var43"] ==0) and (data["var03"]>=1.29149079 and data["var03"] !=0)):
             s.append("1738")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]>=-3.56898403 and data["var43"] !=0) and (data["var06"]<-1.80742383 or data["var06"] ==0) and (data["var22"]<1.55019426 or data["var22"] ==0) and (data["var19"]<-0.968721211 or data["var19"] ==0)):
             s.append("17105")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]>=-3.56898403 and data["var43"] !=0) and (data["var06"]<-1.80742383 or data["var06"] ==0) and (data["var22"]<1.55019426 or data["var22"] ==0) and (data["var19"]>=-0.968721211 and data["var19"] !=0)):
             s.append("17106")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]>=-3.56898403 and data["var43"] !=0) and (data["var06"]<-1.80742383 or data["var06"] ==0) and (data["var22"]>=1.55019426 and data["var22"] !=0)):
             s.append("1768")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]>=-3.56898403 and data["var43"] !=0) and (data["var06"]>=-1.80742383 and data["var06"] !=0) and (data["var47"]<-0.182142332 or data["var47"] ==0) and (data["var10"]<-0.762094557 or data["var10"] ==0)):
             s.append("17107")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]>=-3.56898403 and data["var43"] !=0) and (data["var06"]>=-1.80742383 and data["var06"] !=0) and (data["var47"]<-0.182142332 or data["var47"] ==0) and (data["var10"]>=-0.762094557 and data["var10"] !=0)):
             s.append("17108")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]>=-3.56898403 and data["var43"] !=0) and (data["var06"]>=-1.80742383 and data["var06"] !=0) and (data["var47"]>=-0.182142332 and data["var47"] !=0) and (data["var53"]<0.815483451 or data["var53"] ==0)):
             s.append("17109")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]<-2.35731435 or data["var43"] ==0) and (data["var43"]>=-3.56898403 and data["var43"] !=0) and (data["var06"]>=-1.80742383 and data["var06"] !=0) and (data["var47"]>=-0.182142332 and data["var47"] !=0) and (data["var53"]>=0.815483451 and data["var53"] !=0)):
             s.append("17110")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]<-1.63838351 or data["var56"] ==0) and (data["var25"]<1.10891378 or data["var25"] ==0) and (data["var05"]<-1.39493465 or data["var05"] ==0)):
             s.append("17111")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]<-1.63838351 or data["var56"] ==0) and (data["var25"]<1.10891378 or data["var25"] ==0) and (data["var05"]>=-1.39493465 and data["var05"] !=0)):
             s.append("17112")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]<-1.63838351 or data["var56"] ==0) and (data["var25"]>=1.10891378 and data["var25"] !=0) and (data["var09"]<0.309670746 or data["var09"] ==0)):
             s.append("17113")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]<-1.63838351 or data["var56"] ==0) and (data["var25"]>=1.10891378 and data["var25"] !=0) and (data["var09"]>=0.309670746 and data["var09"] !=0)):
             s.append("17114")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]>=-1.63838351 and data["var56"] !=0) and (data["var08"]<3.51773977 or data["var08"] ==0) and (data["var39"]<-2.71257162 or data["var39"] ==0)):
             s.append("17115")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]>=-1.63838351 and data["var56"] !=0) and (data["var08"]<3.51773977 or data["var08"] ==0) and (data["var39"]>=-2.71257162 and data["var39"] !=0)):
             s.append("17116")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]>=-1.63838351 and data["var56"] !=0) and (data["var08"]>=3.51773977 and data["var08"] !=0) and (data["var17"]<-1.99174368 or data["var17"] ==0)):
             s.append("17117")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]<2.73447466 or data["var10"] ==0) and (data["var56"]>=-1.63838351 and data["var56"] !=0) and (data["var08"]>=3.51773977 and data["var08"] !=0) and (data["var17"]>=-1.99174368 and data["var17"] !=0)):
             s.append("17118")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]>=2.73447466 and data["var10"] !=0) and (data["var26"]<-0.930907726 or data["var26"] ==0) and (data["var14"]<2.11579299 or data["var14"] ==0) and (data["var38"]<2.04682827 or data["var38"] ==0)):
             s.append("17119")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]>=2.73447466 and data["var10"] !=0) and (data["var26"]<-0.930907726 or data["var26"] ==0) and (data["var14"]<2.11579299 or data["var14"] ==0) and (data["var38"]>=2.04682827 and data["var38"] !=0)):
             s.append("17120")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]>=2.73447466 and data["var10"] !=0) and (data["var26"]<-0.930907726 or data["var26"] ==0) and (data["var14"]>=2.11579299 and data["var14"] !=0)):
             s.append("1776")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]>=2.73447466 and data["var10"] !=0) and (data["var26"]>=-0.930907726 and data["var26"] !=0) and (data["var28"]<1.19394743 or data["var28"] ==0) and (data["var50"]<-1.42427588 or data["var50"] ==0)):
             s.append("17121")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]>=2.73447466 and data["var10"] !=0) and (data["var26"]>=-0.930907726 and data["var26"] !=0) and (data["var28"]<1.19394743 or data["var28"] ==0) and (data["var50"]>=-1.42427588 and data["var50"] !=0)):
             s.append("17122")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]>=2.73447466 and data["var10"] !=0) and (data["var26"]>=-0.930907726 and data["var26"] !=0) and (data["var28"]>=1.19394743 and data["var28"] !=0) and (data["var08"]<1.35279322 or data["var08"] ==0)):
             s.append("17123")
    if((data["var16"]<2.40595984 or data["var16"] ==0) and (data["var16"]>=-1.92100489 and data["var16"] !=0) and (data["var43"]>=-2.35731435 and data["var43"] !=0) and (data["var10"]>=2.73447466 and data["var10"] !=0) and (data["var26"]>=-0.930907726 and data["var26"] !=0) and (data["var28"]>=1.19394743 and data["var28"] !=0) and (data["var08"]>=1.35279322 and data["var08"] !=0)):
             s.append("17124")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]<-1.30584335 or data["var51"] ==0) and (data["var42"]<1.62888134 or data["var42"] ==0) and (data["var40"]<-2.20469952 or data["var40"] ==0)):
             s.append("1779")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]<-1.30584335 or data["var51"] ==0) and (data["var42"]<1.62888134 or data["var42"] ==0) and (data["var40"]>=-2.20469952 and data["var40"] !=0) and (data["var03"]<1.62747669 or data["var03"] ==0)):
             s.append("17125")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]<-1.30584335 or data["var51"] ==0) and (data["var42"]<1.62888134 or data["var42"] ==0) and (data["var40"]>=-2.20469952 and data["var40"] !=0) and (data["var03"]>=1.62747669 and data["var03"] !=0)):
             s.append("17126")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]<-1.30584335 or data["var51"] ==0) and (data["var42"]>=1.62888134 and data["var42"] !=0)):
             s.append("1746")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]>=-1.30584335 and data["var51"] !=0) and (data["var06"]<1.95141101 or data["var06"] ==0) and (data["var23"]<1.71108079 or data["var23"] ==0) and (data["var23"]<0.456807256 or data["var23"] ==0)):
             s.append("17127")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]>=-1.30584335 and data["var51"] !=0) and (data["var06"]<1.95141101 or data["var06"] ==0) and (data["var23"]<1.71108079 or data["var23"] ==0) and (data["var23"]>=0.456807256 and data["var23"] !=0)):
             s.append("17128")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]>=-1.30584335 and data["var51"] !=0) and (data["var06"]<1.95141101 or data["var06"] ==0) and (data["var23"]>=1.71108079 and data["var23"] !=0) and (data["var44"]<-0.939685345 or data["var44"] ==0)):
             s.append("17129")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]>=-1.30584335 and data["var51"] !=0) and (data["var06"]<1.95141101 or data["var06"] ==0) and (data["var23"]>=1.71108079 and data["var23"] !=0) and (data["var44"]>=-0.939685345 and data["var44"] !=0)):
             s.append("17130")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]>=-1.30584335 and data["var51"] !=0) and (data["var06"]>=1.95141101 and data["var06"] !=0) and (data["var51"]<-0.502447128 or data["var51"] ==0)):
             s.append("1783")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]<1.32346988 or data["var32"] ==0) and (data["var51"]>=-1.30584335 and data["var51"] !=0) and (data["var06"]>=1.95141101 and data["var06"] !=0) and (data["var51"]>=-0.502447128 and data["var51"] !=0)):
             s.append("1784")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]>=1.32346988 and data["var32"] !=0) and (data["var18"]<-2.28040171 or data["var18"] ==0)):
             s.append("1725")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]>=1.32346988 and data["var32"] !=0) and (data["var18"]>=-2.28040171 and data["var18"] !=0) and (data["var55"]<2.17207336 or data["var55"] ==0) and (data["var28"]<-1.55980575 or data["var28"] ==0)):
             s.append("1785")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]>=1.32346988 and data["var32"] !=0) and (data["var18"]>=-2.28040171 and data["var18"] !=0) and (data["var55"]<2.17207336 or data["var55"] ==0) and (data["var28"]>=-1.55980575 and data["var28"] !=0) and (data["var16"]<2.54445362 or data["var16"] ==0)):
             s.append("17131")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]>=1.32346988 and data["var32"] !=0) and (data["var18"]>=-2.28040171 and data["var18"] !=0) and (data["var55"]<2.17207336 or data["var55"] ==0) and (data["var28"]>=-1.55980575 and data["var28"] !=0) and (data["var16"]>=2.54445362 and data["var16"] !=0)):
             s.append("17132")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]<1.16890192 or data["var43"] ==0) and (data["var32"]>=1.32346988 and data["var32"] !=0) and (data["var18"]>=-2.28040171 and data["var18"] !=0) and (data["var55"]>=2.17207336 and data["var55"] !=0)):
             s.append("1750")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]>=1.16890192 and data["var43"] !=0) and (data["var39"]<-1.91187024 or data["var39"] ==0)):
             s.append("1713")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]>=1.16890192 and data["var43"] !=0) and (data["var39"]>=-1.91187024 and data["var39"] !=0) and (data["var34"]<1.46618056 or data["var34"] ==0) and (data["var45"]<-1.41088068 or data["var45"] ==0) and (data["var11"]<0.167955801 or data["var11"] ==0)):
             s.append("1787")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]>=1.16890192 and data["var43"] !=0) and (data["var39"]>=-1.91187024 and data["var39"] !=0) and (data["var34"]<1.46618056 or data["var34"] ==0) and (data["var45"]<-1.41088068 or data["var45"] ==0) and (data["var11"]>=0.167955801 and data["var11"] !=0)):
             s.append("1788")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]>=1.16890192 and data["var43"] !=0) and (data["var39"]>=-1.91187024 and data["var39"] !=0) and (data["var34"]<1.46618056 or data["var34"] ==0) and (data["var45"]>=-1.41088068 and data["var45"] !=0)):
             s.append("1752")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]>=1.16890192 and data["var43"] !=0) and (data["var39"]>=-1.91187024 and data["var39"] !=0) and (data["var34"]>=1.46618056 and data["var34"] !=0) and (data["var27"]<-0.641125381 or data["var27"] ==0)):
             s.append("1753")
    if((data["var16"]>=2.40595984 and data["var16"] !=0) and (data["var43"]>=1.16890192 and data["var43"] !=0) and (data["var39"]>=-1.91187024 and data["var39"] !=0) and (data["var34"]>=1.46618056 and data["var34"] !=0) and (data["var27"]>=-0.641125381 and data["var27"] !=0)):
             s.append("1754")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]<-0.687546074 or data["var30"] ==0) and (data["var51"]<-0.0659005493 or data["var51"] ==0) and (data["var12"]<0.10948801 or data["var12"] ==0) and (data["var02"]<0.547424555 or data["var02"] ==0)):
             s.append("1861")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]<-0.687546074 or data["var30"] ==0) and (data["var51"]<-0.0659005493 or data["var51"] ==0) and (data["var12"]<0.10948801 or data["var12"] ==0) and (data["var02"]>=0.547424555 and data["var02"] !=0)):
             s.append("1862")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]<-0.687546074 or data["var30"] ==0) and (data["var51"]<-0.0659005493 or data["var51"] ==0) and (data["var12"]>=0.10948801 and data["var12"] !=0) and (data["var13"]<-0.322774649 or data["var13"] ==0)):
             s.append("1863")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]<-0.687546074 or data["var30"] ==0) and (data["var51"]<-0.0659005493 or data["var51"] ==0) and (data["var12"]>=0.10948801 and data["var12"] !=0) and (data["var13"]>=-0.322774649 and data["var13"] !=0)):
             s.append("1864")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]<-0.687546074 or data["var30"] ==0) and (data["var51"]>=-0.0659005493 and data["var51"] !=0) and (data["var05"]<1.39995623 or data["var05"] ==0) and (data["var41"]<-1.47878289 or data["var41"] ==0)):
             s.append("1865")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]<-0.687546074 or data["var30"] ==0) and (data["var51"]>=-0.0659005493 and data["var51"] !=0) and (data["var05"]<1.39995623 or data["var05"] ==0) and (data["var41"]>=-1.47878289 and data["var41"] !=0)):
             s.append("1866")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]<-0.687546074 or data["var30"] ==0) and (data["var51"]>=-0.0659005493 and data["var51"] !=0) and (data["var05"]>=1.39995623 and data["var05"] !=0)):
             s.append("1834")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]>=-0.687546074 and data["var30"] !=0) and (data["var46"]<1.56522751 or data["var46"] ==0) and (data["var13"]<2.36086845 or data["var13"] ==0) and (data["var58"]<-2.60662246 or data["var58"] ==0)):
             s.append("1867")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]>=-0.687546074 and data["var30"] !=0) and (data["var46"]<1.56522751 or data["var46"] ==0) and (data["var13"]<2.36086845 or data["var13"] ==0) and (data["var58"]>=-2.60662246 and data["var58"] !=0) and (data["var04"]<-2.65163088 or data["var04"] ==0)):
             s.append("18109")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]>=-0.687546074 and data["var30"] !=0) and (data["var46"]<1.56522751 or data["var46"] ==0) and (data["var13"]<2.36086845 or data["var13"] ==0) and (data["var58"]>=-2.60662246 and data["var58"] !=0) and (data["var04"]>=-2.65163088 and data["var04"] !=0)):
             s.append("18110")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]>=-0.687546074 and data["var30"] !=0) and (data["var46"]<1.56522751 or data["var46"] ==0) and (data["var13"]>=2.36086845 and data["var13"] !=0)):
             s.append("1836")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]>=-0.687546074 and data["var30"] !=0) and (data["var46"]>=1.56522751 and data["var46"] !=0) and (data["var10"]<0.400862247 or data["var10"] ==0) and (data["var13"]<0.141591951 or data["var13"] ==0)):
             s.append("1869")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]>=-0.687546074 and data["var30"] !=0) and (data["var46"]>=1.56522751 and data["var46"] !=0) and (data["var10"]<0.400862247 or data["var10"] ==0) and (data["var13"]>=0.141591951 and data["var13"] !=0)):
             s.append("1870")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]<-0.566646338 or data["var26"] ==0) and (data["var30"]>=-0.687546074 and data["var30"] !=0) and (data["var46"]>=1.56522751 and data["var46"] !=0) and (data["var10"]>=0.400862247 and data["var10"] !=0)):
             s.append("1838")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]<-0.996388555 or data["var06"] ==0) and (data["var44"]<0.749672532 or data["var44"] ==0) and (data["var08"]<-2.38577986 or data["var08"] ==0)):
             s.append("18111")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]<-0.996388555 or data["var06"] ==0) and (data["var44"]<0.749672532 or data["var44"] ==0) and (data["var08"]>=-2.38577986 and data["var08"] !=0)):
             s.append("18112")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]<-0.996388555 or data["var06"] ==0) and (data["var44"]>=0.749672532 and data["var44"] !=0) and (data["var22"]<-0.526579499 or data["var22"] ==0)):
             s.append("18113")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]<-0.996388555 or data["var06"] ==0) and (data["var44"]>=0.749672532 and data["var44"] !=0) and (data["var22"]>=-0.526579499 and data["var22"] !=0)):
             s.append("18114")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]>=-0.996388555 and data["var06"] !=0) and (data["var42"]<-0.756190538 or data["var42"] ==0) and (data["var08"]<-2.38293505 or data["var08"] ==0)):
             s.append("18115")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]>=-0.996388555 and data["var06"] !=0) and (data["var42"]<-0.756190538 or data["var42"] ==0) and (data["var08"]>=-2.38293505 and data["var08"] !=0)):
             s.append("18116")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]>=-0.996388555 and data["var06"] !=0) and (data["var42"]>=-0.756190538 and data["var42"] !=0) and (data["var51"]<0.396502733 or data["var51"] ==0)):
             s.append("18117")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]<1.46511984 or data["var06"] ==0) and (data["var06"]>=-0.996388555 and data["var06"] !=0) and (data["var42"]>=-0.756190538 and data["var42"] !=0) and (data["var51"]>=0.396502733 and data["var51"] !=0)):
             s.append("18118")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]>=1.46511984 and data["var06"] !=0) and (data["var28"]<0.53732729 or data["var28"] ==0)):
             s.append("1841")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]>=1.46511984 and data["var06"] !=0) and (data["var28"]>=0.53732729 and data["var28"] !=0) and (data["var14"]<-0.497849762 or data["var14"] ==0)):
             s.append("1875")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]<1.14073277 or data["var12"] ==0) and (data["var06"]>=1.46511984 and data["var06"] !=0) and (data["var28"]>=0.53732729 and data["var28"] !=0) and (data["var14"]>=-0.497849762 and data["var14"] !=0)):
             s.append("1876")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]>=1.14073277 and data["var12"] !=0) and (data["var30"]<-0.970644474 or data["var30"] ==0) and (data["var22"]<-0.357085288 or data["var22"] ==0)):
             s.append("1843")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]>=1.14073277 and data["var12"] !=0) and (data["var30"]<-0.970644474 or data["var30"] ==0) and (data["var22"]>=-0.357085288 and data["var22"] !=0) and (data["var40"]<-1.12821007 or data["var40"] ==0)):
             s.append("1877")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]>=1.14073277 and data["var12"] !=0) and (data["var30"]<-0.970644474 or data["var30"] ==0) and (data["var22"]>=-0.357085288 and data["var22"] !=0) and (data["var40"]>=-1.12821007 and data["var40"] !=0)):
             s.append("1878")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]>=1.14073277 and data["var12"] !=0) and (data["var30"]>=-0.970644474 and data["var30"] !=0) and (data["var09"]<-2.47121239 or data["var09"] ==0)):
             s.append("1845")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]>=1.14073277 and data["var12"] !=0) and (data["var30"]>=-0.970644474 and data["var30"] !=0) and (data["var09"]>=-2.47121239 and data["var09"] !=0) and (data["var56"]<1.28362679 or data["var56"] ==0) and (data["var45"]<-1.07675815 or data["var45"] ==0)):
             s.append("18119")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]>=1.14073277 and data["var12"] !=0) and (data["var30"]>=-0.970644474 and data["var30"] !=0) and (data["var09"]>=-2.47121239 and data["var09"] !=0) and (data["var56"]<1.28362679 or data["var56"] ==0) and (data["var45"]>=-1.07675815 and data["var45"] !=0)):
             s.append("18120")
    if((data["var08"]<-2.18617725 or data["var08"] ==0) and (data["var26"]>=-0.566646338 and data["var26"] !=0) and (data["var12"]>=1.14073277 and data["var12"] !=0) and (data["var30"]>=-0.970644474 and data["var30"] !=0) and (data["var09"]>=-2.47121239 and data["var09"] !=0) and (data["var56"]>=1.28362679 and data["var56"] !=0)):
             s.append("1880")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]<-1.46863115 or data["var45"] ==0) and (data["var13"]<-1.53004062 or data["var13"] ==0)):
             s.append("1823")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]<-1.46863115 or data["var45"] ==0) and (data["var13"]>=-1.53004062 and data["var13"] !=0) and (data["var07"]<-1.3517251 or data["var07"] ==0) and (data["var12"]<-0.0152660049 or data["var12"] ==0)):
             s.append("1881")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]<-1.46863115 or data["var45"] ==0) and (data["var13"]>=-1.53004062 and data["var13"] !=0) and (data["var07"]<-1.3517251 or data["var07"] ==0) and (data["var12"]>=-0.0152660049 and data["var12"] !=0)):
             s.append("1882")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]<-1.46863115 or data["var45"] ==0) and (data["var13"]>=-1.53004062 and data["var13"] !=0) and (data["var07"]>=-1.3517251 and data["var07"] !=0) and (data["var38"]<1.51062882 or data["var38"] ==0) and (data["var44"]<-1.18242991 or data["var44"] ==0)):
             s.append("18121")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]<-1.46863115 or data["var45"] ==0) and (data["var13"]>=-1.53004062 and data["var13"] !=0) and (data["var07"]>=-1.3517251 and data["var07"] !=0) and (data["var38"]<1.51062882 or data["var38"] ==0) and (data["var44"]>=-1.18242991 and data["var44"] !=0)):
             s.append("18122")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]<-1.46863115 or data["var45"] ==0) and (data["var13"]>=-1.53004062 and data["var13"] !=0) and (data["var07"]>=-1.3517251 and data["var07"] !=0) and (data["var38"]>=1.51062882 and data["var38"] !=0)):
             s.append("1884")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]<0.022455046 or data["var57"] ==0) and (data["var48"]<-0.0190323032 or data["var48"] ==0) and (data["var06"]<1.55013382 or data["var06"] ==0)):
             s.append("18123")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]<0.022455046 or data["var57"] ==0) and (data["var48"]<-0.0190323032 or data["var48"] ==0) and (data["var06"]>=1.55013382 and data["var06"] !=0)):
             s.append("18124")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]<0.022455046 or data["var57"] ==0) and (data["var48"]>=-0.0190323032 and data["var48"] !=0) and (data["var06"]<-0.26703909 or data["var06"] ==0)):
             s.append("18125")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]<0.022455046 or data["var57"] ==0) and (data["var48"]>=-0.0190323032 and data["var48"] !=0) and (data["var06"]>=-0.26703909 and data["var06"] !=0)):
             s.append("18126")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]>=0.022455046 and data["var57"] !=0) and (data["var43"]<0.805039525 or data["var43"] ==0) and (data["var02"]<-0.00320245698 or data["var02"] ==0)):
             s.append("18127")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]>=0.022455046 and data["var57"] !=0) and (data["var43"]<0.805039525 or data["var43"] ==0) and (data["var02"]>=-0.00320245698 and data["var02"] !=0)):
             s.append("18128")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]>=0.022455046 and data["var57"] !=0) and (data["var43"]>=0.805039525 and data["var43"] !=0) and (data["var22"]<1.10910571 or data["var22"] ==0)):
             s.append("18129")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]<0.764971733 or data["var34"] ==0) and (data["var57"]>=0.022455046 and data["var57"] !=0) and (data["var43"]>=0.805039525 and data["var43"] !=0) and (data["var22"]>=1.10910571 and data["var22"] !=0)):
             s.append("18130")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]>=0.764971733 and data["var34"] !=0) and (data["var40"]<-0.165397614 or data["var40"] ==0) and (data["var57"]<0.548115373 or data["var57"] ==0) and (data["var06"]<-0.423110247 or data["var06"] ==0)):
             s.append("18131")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]>=0.764971733 and data["var34"] !=0) and (data["var40"]<-0.165397614 or data["var40"] ==0) and (data["var57"]<0.548115373 or data["var57"] ==0) and (data["var06"]>=-0.423110247 and data["var06"] !=0)):
             s.append("18132")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]>=0.764971733 and data["var34"] !=0) and (data["var40"]<-0.165397614 or data["var40"] ==0) and (data["var57"]>=0.548115373 and data["var57"] !=0)):
             s.append("1890")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]>=0.764971733 and data["var34"] !=0) and (data["var40"]>=-0.165397614 and data["var40"] !=0) and (data["var25"]<-0.534727275 or data["var25"] ==0) and (data["var24"]<-0.824732244 or data["var24"] ==0)):
             s.append("18133")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]>=0.764971733 and data["var34"] !=0) and (data["var40"]>=-0.165397614 and data["var40"] !=0) and (data["var25"]<-0.534727275 or data["var25"] ==0) and (data["var24"]>=-0.824732244 and data["var24"] !=0)):
             s.append("18134")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]>=0.764971733 and data["var34"] !=0) and (data["var40"]>=-0.165397614 and data["var40"] !=0) and (data["var25"]>=-0.534727275 and data["var25"] !=0) and (data["var26"]<-1.77661371 or data["var26"] ==0)):
             s.append("18135")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]<-2.32543755 or data["var58"] ==0) and (data["var45"]>=-1.46863115 and data["var45"] !=0) and (data["var34"]>=0.764971733 and data["var34"] !=0) and (data["var40"]>=-0.165397614 and data["var40"] !=0) and (data["var25"]>=-0.534727275 and data["var25"] !=0) and (data["var26"]>=-1.77661371 and data["var26"] !=0)):
             s.append("18136")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]<-2.75344276 or data["var49"] ==0) and (data["var33"]<0.8247751 or data["var33"] ==0) and (data["var13"]<-0.320261627 or data["var13"] ==0)):
             s.append("18137")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]<-2.75344276 or data["var49"] ==0) and (data["var33"]<0.8247751 or data["var33"] ==0) and (data["var13"]>=-0.320261627 and data["var13"] !=0)):
             s.append("18138")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]<-2.75344276 or data["var49"] ==0) and (data["var33"]>=0.8247751 and data["var33"] !=0) and (data["var47"]<1.04848719 or data["var47"] ==0)):
             s.append("18139")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]<-2.75344276 or data["var49"] ==0) and (data["var33"]>=0.8247751 and data["var33"] !=0) and (data["var47"]>=1.04848719 and data["var47"] !=0)):
             s.append("18140")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]>=-2.75344276 and data["var49"] !=0) and (data["var23"]<1.46977234 or data["var23"] ==0) and (data["var51"]<-2.56169009 or data["var51"] ==0)):
             s.append("18141")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]>=-2.75344276 and data["var49"] !=0) and (data["var23"]<1.46977234 or data["var23"] ==0) and (data["var51"]>=-2.56169009 and data["var51"] !=0)):
             s.append("18142")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]>=-2.75344276 and data["var49"] !=0) and (data["var23"]>=1.46977234 and data["var23"] !=0) and (data["var23"]<2.94108391 or data["var23"] ==0)):
             s.append("18143")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]<1.65371251 or data["var57"] ==0) and (data["var49"]>=-2.75344276 and data["var49"] !=0) and (data["var23"]>=1.46977234 and data["var23"] !=0) and (data["var23"]>=2.94108391 and data["var23"] !=0)):
             s.append("18144")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]>=1.65371251 and data["var57"] !=0) and (data["var08"]<1.67481863 or data["var08"] ==0) and (data["var10"]<-1.67518532 or data["var10"] ==0) and (data["var37"]<-0.101018429 or data["var37"] ==0)):
             s.append("18145")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]>=1.65371251 and data["var57"] !=0) and (data["var08"]<1.67481863 or data["var08"] ==0) and (data["var10"]<-1.67518532 or data["var10"] ==0) and (data["var37"]>=-0.101018429 and data["var37"] !=0)):
             s.append("18146")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]>=1.65371251 and data["var57"] !=0) and (data["var08"]<1.67481863 or data["var08"] ==0) and (data["var10"]>=-1.67518532 and data["var10"] !=0) and (data["var35"]<-2.38694501 or data["var35"] ==0)):
             s.append("18147")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]>=1.65371251 and data["var57"] !=0) and (data["var08"]<1.67481863 or data["var08"] ==0) and (data["var10"]>=-1.67518532 and data["var10"] !=0) and (data["var35"]>=-2.38694501 and data["var35"] !=0)):
             s.append("18148")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]>=1.65371251 and data["var57"] !=0) and (data["var08"]>=1.67481863 and data["var08"] !=0) and (data["var42"]<-1.96639299 or data["var42"] ==0)):
             s.append("1899")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]>=1.65371251 and data["var57"] !=0) and (data["var08"]>=1.67481863 and data["var08"] !=0) and (data["var42"]>=-1.96639299 and data["var42"] !=0) and (data["var03"]<-0.747047007 or data["var03"] ==0)):
             s.append("18149")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]<2.74508071 or data["var11"] ==0) and (data["var57"]>=1.65371251 and data["var57"] !=0) and (data["var08"]>=1.67481863 and data["var08"] !=0) and (data["var42"]>=-1.96639299 and data["var42"] !=0) and (data["var03"]>=-0.747047007 and data["var03"] !=0)):
             s.append("18150")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]<0.273734808 or data["var37"] ==0) and (data["var48"]<0.580080271 or data["var48"] ==0) and (data["var26"]<2.16568255 or data["var26"] ==0) and (data["var32"]<-2.70125484 or data["var32"] ==0)):
             s.append("18151")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]<0.273734808 or data["var37"] ==0) and (data["var48"]<0.580080271 or data["var48"] ==0) and (data["var26"]<2.16568255 or data["var26"] ==0) and (data["var32"]>=-2.70125484 and data["var32"] !=0)):
             s.append("18152")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]<0.273734808 or data["var37"] ==0) and (data["var48"]<0.580080271 or data["var48"] ==0) and (data["var26"]>=2.16568255 and data["var26"] !=0)):
             s.append("18102")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]<0.273734808 or data["var37"] ==0) and (data["var48"]>=0.580080271 and data["var48"] !=0) and (data["var48"]<0.760430932 or data["var48"] ==0)):
             s.append("18103")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]<0.273734808 or data["var37"] ==0) and (data["var48"]>=0.580080271 and data["var48"] !=0) and (data["var48"]>=0.760430932 and data["var48"] !=0) and (data["var03"]<-0.1323369 or data["var03"] ==0)):
             s.append("18153")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]<0.273734808 or data["var37"] ==0) and (data["var48"]>=0.580080271 and data["var48"] !=0) and (data["var48"]>=0.760430932 and data["var48"] !=0) and (data["var03"]>=-0.1323369 and data["var03"] !=0)):
             s.append("18154")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]>=0.273734808 and data["var37"] !=0) and (data["var11"]<3.55913591 or data["var11"] ==0) and (data["var23"]<-1.20348406 or data["var23"] ==0) and (data["var56"]<-1.42654824 or data["var56"] ==0)):
             s.append("18155")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]>=0.273734808 and data["var37"] !=0) and (data["var11"]<3.55913591 or data["var11"] ==0) and (data["var23"]<-1.20348406 or data["var23"] ==0) and (data["var56"]>=-1.42654824 and data["var56"] !=0)):
             s.append("18156")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]>=0.273734808 and data["var37"] !=0) and (data["var11"]<3.55913591 or data["var11"] ==0) and (data["var23"]>=-1.20348406 and data["var23"] !=0) and (data["var39"]<-0.140771598 or data["var39"] ==0)):
             s.append("18157")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]>=0.273734808 and data["var37"] !=0) and (data["var11"]<3.55913591 or data["var11"] ==0) and (data["var23"]>=-1.20348406 and data["var23"] !=0) and (data["var39"]>=-0.140771598 and data["var39"] !=0)):
             s.append("18158")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]>=0.273734808 and data["var37"] !=0) and (data["var11"]>=3.55913591 and data["var11"] !=0) and (data["var31"]<1.5205698 or data["var31"] ==0) and (data["var26"]<1.54629517 or data["var26"] ==0)):
             s.append("18159")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]>=0.273734808 and data["var37"] !=0) and (data["var11"]>=3.55913591 and data["var11"] !=0) and (data["var31"]<1.5205698 or data["var31"] ==0) and (data["var26"]>=1.54629517 and data["var26"] !=0)):
             s.append("18160")
    if((data["var08"]>=-2.18617725 and data["var08"] !=0) and (data["var58"]>=-2.32543755 and data["var58"] !=0) and (data["var11"]>=2.74508071 and data["var11"] !=0) and (data["var37"]>=0.273734808 and data["var37"] !=0) and (data["var11"]>=3.55913591 and data["var11"] !=0) and (data["var31"]>=1.5205698 and data["var31"] !=0)):
             s.append("18108")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<2.10046721 or data["var17"] ==0) and (data["var40"]<1.63628232 or data["var40"] ==0) and (data["var50"]<-0.717072964 or data["var50"] ==0) and (data["var50"]<-0.981737196 or data["var50"] ==0) and (data["var21"]<-0.0183663405 or data["var21"] ==0)):
             s.append("19101")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<2.10046721 or data["var17"] ==0) and (data["var40"]<1.63628232 or data["var40"] ==0) and (data["var50"]<-0.717072964 or data["var50"] ==0) and (data["var50"]<-0.981737196 or data["var50"] ==0) and (data["var21"]>=-0.0183663405 and data["var21"] !=0)):
             s.append("19102")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<2.10046721 or data["var17"] ==0) and (data["var40"]<1.63628232 or data["var40"] ==0) and (data["var50"]<-0.717072964 or data["var50"] ==0) and (data["var50"]>=-0.981737196 and data["var50"] !=0)):
             s.append("1958")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<2.10046721 or data["var17"] ==0) and (data["var40"]<1.63628232 or data["var40"] ==0) and (data["var50"]>=-0.717072964 and data["var50"] !=0) and (data["var24"]<-2.04592109 or data["var24"] ==0)):
             s.append("1959")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<2.10046721 or data["var17"] ==0) and (data["var40"]<1.63628232 or data["var40"] ==0) and (data["var50"]>=-0.717072964 and data["var50"] !=0) and (data["var24"]>=-2.04592109 and data["var24"] !=0)):
             s.append("1960")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<2.10046721 or data["var17"] ==0) and (data["var40"]>=1.63628232 and data["var40"] !=0) and (data["var12"]<-1.00311494 or data["var12"] ==0)):
             s.append("1933")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]<2.10046721 or data["var17"] ==0) and (data["var40"]>=1.63628232 and data["var40"] !=0) and (data["var12"]>=-1.00311494 and data["var12"] !=0)):
             s.append("1934")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]>=2.10046721 and data["var17"] !=0) and (data["var13"]<-0.0612075701 or data["var13"] ==0)):
             s.append("1917")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]<-0.990586758 or data["var42"] ==0) and (data["var17"]>=2.10046721 and data["var17"] !=0) and (data["var13"]>=-0.0612075701 and data["var13"] !=0)):
             s.append("1918")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]<-1.2395215 or data["var21"] ==0) and (data["var36"]<1.03854561 or data["var36"] ==0) and (data["var43"]<-0.717691541 or data["var43"] ==0) and (data["var41"]<0.627283573 or data["var41"] ==0) and (data["var18"]<0.0362031944 or data["var18"] ==0)):
             s.append("19103")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]<-1.2395215 or data["var21"] ==0) and (data["var36"]<1.03854561 or data["var36"] ==0) and (data["var43"]<-0.717691541 or data["var43"] ==0) and (data["var41"]<0.627283573 or data["var41"] ==0) and (data["var18"]>=0.0362031944 and data["var18"] !=0)):
             s.append("19104")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]<-1.2395215 or data["var21"] ==0) and (data["var36"]<1.03854561 or data["var36"] ==0) and (data["var43"]<-0.717691541 or data["var43"] ==0) and (data["var41"]>=0.627283573 and data["var41"] !=0)):
             s.append("1962")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]<-1.2395215 or data["var21"] ==0) and (data["var36"]<1.03854561 or data["var36"] ==0) and (data["var43"]>=-0.717691541 and data["var43"] !=0) and (data["var32"]<-1.62761378 or data["var32"] ==0)):
             s.append("1963")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]<-1.2395215 or data["var21"] ==0) and (data["var36"]<1.03854561 or data["var36"] ==0) and (data["var43"]>=-0.717691541 and data["var43"] !=0) and (data["var32"]>=-1.62761378 and data["var32"] !=0)):
             s.append("1964")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]<-1.2395215 or data["var21"] ==0) and (data["var36"]>=1.03854561 and data["var36"] !=0) and (data["var27"]<0.719472766 or data["var27"] ==0)):
             s.append("1937")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]<-1.2395215 or data["var21"] ==0) and (data["var36"]>=1.03854561 and data["var36"] !=0) and (data["var27"]>=0.719472766 and data["var27"] !=0)):
             s.append("1938")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]<-0.478682995 or data["var33"] ==0) and (data["var27"]<1.66062331 or data["var27"] ==0) and (data["var21"]<1.17943811 or data["var21"] ==0) and (data["var13"]<-1.79337394 or data["var13"] ==0)):
             s.append("19105")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]<-0.478682995 or data["var33"] ==0) and (data["var27"]<1.66062331 or data["var27"] ==0) and (data["var21"]<1.17943811 or data["var21"] ==0) and (data["var13"]>=-1.79337394 and data["var13"] !=0)):
             s.append("19106")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]<-0.478682995 or data["var33"] ==0) and (data["var27"]<1.66062331 or data["var27"] ==0) and (data["var21"]>=1.17943811 and data["var21"] !=0) and (data["var37"]<0.096240297 or data["var37"] ==0)):
             s.append("19107")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]<-0.478682995 or data["var33"] ==0) and (data["var27"]<1.66062331 or data["var27"] ==0) and (data["var21"]>=1.17943811 and data["var21"] !=0) and (data["var37"]>=0.096240297 and data["var37"] !=0)):
             s.append("19108")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]<-0.478682995 or data["var33"] ==0) and (data["var27"]>=1.66062331 and data["var27"] !=0) and (data["var59"]<0.265003085 or data["var59"] ==0)):
             s.append("1967")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]<-0.478682995 or data["var33"] ==0) and (data["var27"]>=1.66062331 and data["var27"] !=0) and (data["var59"]>=0.265003085 and data["var59"] !=0)):
             s.append("1968")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]<0.136167675 or data["var42"] ==0) and (data["var37"]<-0.963949442 or data["var37"] ==0) and (data["var26"]<1.14038765 or data["var26"] ==0)):
             s.append("19109")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]<0.136167675 or data["var42"] ==0) and (data["var37"]<-0.963949442 or data["var37"] ==0) and (data["var26"]>=1.14038765 and data["var26"] !=0)):
             s.append("19110")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]<0.136167675 or data["var42"] ==0) and (data["var37"]>=-0.963949442 and data["var37"] !=0) and (data["var15"]<0.556078672 or data["var15"] ==0)):
             s.append("19111")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]<0.136167675 or data["var42"] ==0) and (data["var37"]>=-0.963949442 and data["var37"] !=0) and (data["var15"]>=0.556078672 and data["var15"] !=0)):
             s.append("19112")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]>=0.136167675 and data["var42"] !=0) and (data["var04"]<-1.32073629 or data["var04"] ==0) and (data["var04"]<-1.74343169 or data["var04"] ==0)):
             s.append("19113")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]>=0.136167675 and data["var42"] !=0) and (data["var04"]<-1.32073629 or data["var04"] ==0) and (data["var04"]>=-1.74343169 and data["var04"] !=0)):
             s.append("19114")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]>=0.136167675 and data["var42"] !=0) and (data["var04"]>=-1.32073629 and data["var04"] !=0) and (data["var01"]<1.44263494 or data["var01"] ==0)):
             s.append("19115")
    if((data["var03"]<-2.21224356 or data["var03"] ==0) and (data["var42"]>=-0.990586758 and data["var42"] !=0) and (data["var21"]>=-1.2395215 and data["var21"] !=0) and (data["var33"]>=-0.478682995 and data["var33"] !=0) and (data["var42"]>=0.136167675 and data["var42"] !=0) and (data["var04"]>=-1.32073629 and data["var04"] !=0) and (data["var01"]>=1.44263494 and data["var01"] !=0)):
             s.append("19116")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]<-1.12938857 or data["var46"] ==0) and (data["var56"]<1.81894422 or data["var56"] ==0) and (data["var22"]<0.603633046 or data["var22"] ==0) and (data["var26"]<1.98800898 or data["var26"] ==0)):
             s.append("19117")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]<-1.12938857 or data["var46"] ==0) and (data["var56"]<1.81894422 or data["var56"] ==0) and (data["var22"]<0.603633046 or data["var22"] ==0) and (data["var26"]>=1.98800898 and data["var26"] !=0)):
             s.append("19118")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]<-1.12938857 or data["var46"] ==0) and (data["var56"]<1.81894422 or data["var56"] ==0) and (data["var22"]>=0.603633046 and data["var22"] !=0) and (data["var23"]<1.63129032 or data["var23"] ==0)):
             s.append("19119")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]<-1.12938857 or data["var46"] ==0) and (data["var56"]<1.81894422 or data["var56"] ==0) and (data["var22"]>=0.603633046 and data["var22"] !=0) and (data["var23"]>=1.63129032 and data["var23"] !=0)):
             s.append("19120")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]<-1.12938857 or data["var46"] ==0) and (data["var56"]>=1.81894422 and data["var56"] !=0) and (data["var42"]<-0.0230437368 or data["var42"] ==0)):
             s.append("1975")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]<-1.12938857 or data["var46"] ==0) and (data["var56"]>=1.81894422 and data["var56"] !=0) and (data["var42"]>=-0.0230437368 and data["var42"] !=0) and (data["var27"]<-0.179113835 or data["var27"] ==0)):
             s.append("19121")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]<-1.12938857 or data["var46"] ==0) and (data["var56"]>=1.81894422 and data["var56"] !=0) and (data["var42"]>=-0.0230437368 and data["var42"] !=0) and (data["var27"]>=-0.179113835 and data["var27"] !=0)):
             s.append("19122")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]<1.3112638 or data["var19"] ==0) and (data["var59"]<-1.85096347 or data["var59"] ==0) and (data["var51"]<1.49717712 or data["var51"] ==0)):
             s.append("19123")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]<1.3112638 or data["var19"] ==0) and (data["var59"]<-1.85096347 or data["var59"] ==0) and (data["var51"]>=1.49717712 and data["var51"] !=0)):
             s.append("19124")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]<1.3112638 or data["var19"] ==0) and (data["var59"]>=-1.85096347 and data["var59"] !=0) and (data["var27"]<-0.647692919 or data["var27"] ==0)):
             s.append("19125")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]<1.3112638 or data["var19"] ==0) and (data["var59"]>=-1.85096347 and data["var59"] !=0) and (data["var27"]>=-0.647692919 and data["var27"] !=0)):
             s.append("19126")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]>=1.3112638 and data["var19"] !=0) and (data["var16"]<-1.12420189 or data["var16"] ==0) and (data["var14"]<-1.11039257 or data["var14"] ==0)):
             s.append("19127")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]>=1.3112638 and data["var19"] !=0) and (data["var16"]<-1.12420189 or data["var16"] ==0) and (data["var14"]>=-1.11039257 and data["var14"] !=0)):
             s.append("19128")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]>=1.3112638 and data["var19"] !=0) and (data["var16"]>=-1.12420189 and data["var16"] !=0) and (data["var56"]<0.229832843 or data["var56"] ==0)):
             s.append("19129")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]<-1.58177781 or data["var17"] ==0) and (data["var46"]>=-1.12938857 and data["var46"] !=0) and (data["var19"]>=1.3112638 and data["var19"] !=0) and (data["var16"]>=-1.12420189 and data["var16"] !=0) and (data["var56"]>=0.229832843 and data["var56"] !=0)):
             s.append("19130")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]<-2.32771826 or data["var19"] ==0) and (data["var16"]<0.681451917 or data["var16"] ==0) and (data["var28"]<-0.865724623 or data["var28"] ==0)):
             s.append("19131")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]<-2.32771826 or data["var19"] ==0) and (data["var16"]<0.681451917 or data["var16"] ==0) and (data["var28"]>=-0.865724623 and data["var28"] !=0)):
             s.append("19132")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]<-2.32771826 or data["var19"] ==0) and (data["var16"]>=0.681451917 and data["var16"] !=0) and (data["var06"]<-0.536635399 or data["var06"] ==0)):
             s.append("19133")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]<-2.32771826 or data["var19"] ==0) and (data["var16"]>=0.681451917 and data["var16"] !=0) and (data["var06"]>=-0.536635399 and data["var06"] !=0)):
             s.append("19134")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]>=-2.32771826 and data["var19"] !=0) and (data["var10"]<-2.23182106 or data["var10"] ==0) and (data["var26"]<-1.0864116 or data["var26"] ==0)):
             s.append("19135")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]>=-2.32771826 and data["var19"] !=0) and (data["var10"]<-2.23182106 or data["var10"] ==0) and (data["var26"]>=-1.0864116 and data["var26"] !=0)):
             s.append("19136")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]>=-2.32771826 and data["var19"] !=0) and (data["var10"]>=-2.23182106 and data["var10"] !=0) and (data["var58"]<2.41049385 or data["var58"] ==0)):
             s.append("19137")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]<1.79385257 or data["var55"] ==0) and (data["var19"]>=-2.32771826 and data["var19"] !=0) and (data["var10"]>=-2.23182106 and data["var10"] !=0) and (data["var58"]>=2.41049385 and data["var58"] !=0)):
             s.append("19138")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]<3.22838092 or data["var55"] ==0) and (data["var21"]<-1.82993793 or data["var21"] ==0) and (data["var07"]<2.2939198 or data["var07"] ==0)):
             s.append("19139")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]<3.22838092 or data["var55"] ==0) and (data["var21"]<-1.82993793 or data["var21"] ==0) and (data["var07"]>=2.2939198 and data["var07"] !=0)):
             s.append("19140")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]<3.22838092 or data["var55"] ==0) and (data["var21"]>=-1.82993793 and data["var21"] !=0) and (data["var12"]<2.11683774 or data["var12"] ==0)):
             s.append("19141")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]<3.22838092 or data["var55"] ==0) and (data["var21"]>=-1.82993793 and data["var21"] !=0) and (data["var12"]>=2.11683774 and data["var12"] !=0)):
             s.append("19142")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]>=3.22838092 and data["var55"] !=0) and (data["var47"]<-0.993337393 or data["var47"] ==0) and (data["var55"]<3.54021502 or data["var55"] ==0)):
             s.append("19143")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]>=3.22838092 and data["var55"] !=0) and (data["var47"]<-0.993337393 or data["var47"] ==0) and (data["var55"]>=3.54021502 and data["var55"] !=0)):
             s.append("19144")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]>=3.22838092 and data["var55"] !=0) and (data["var47"]>=-0.993337393 and data["var47"] !=0) and (data["var19"]<1.76236129 or data["var19"] ==0)):
             s.append("19145")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]<1.97562468 or data["var17"] ==0) and (data["var17"]>=-1.58177781 and data["var17"] !=0) and (data["var55"]>=1.79385257 and data["var55"] !=0) and (data["var55"]>=3.22838092 and data["var55"] !=0) and (data["var47"]>=-0.993337393 and data["var47"] !=0) and (data["var19"]>=1.76236129 and data["var19"] !=0)):
             s.append("19146")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]<2.35520577 or data["var45"] ==0) and (data["var43"]<-0.00820540264 or data["var43"] ==0) and (data["var05"]<2.00932693 or data["var05"] ==0) and (data["var19"]<-1.68871975 or data["var19"] ==0)):
             s.append("19147")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]<2.35520577 or data["var45"] ==0) and (data["var43"]<-0.00820540264 or data["var43"] ==0) and (data["var05"]<2.00932693 or data["var05"] ==0) and (data["var19"]>=-1.68871975 and data["var19"] !=0)):
             s.append("19148")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]<2.35520577 or data["var45"] ==0) and (data["var43"]<-0.00820540264 or data["var43"] ==0) and (data["var05"]>=2.00932693 and data["var05"] !=0)):
             s.append("1990")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]<2.35520577 or data["var45"] ==0) and (data["var43"]>=-0.00820540264 and data["var43"] !=0) and (data["var56"]<-1.92028618 or data["var56"] ==0) and (data["var36"]<1.03473449 or data["var36"] ==0)):
             s.append("19149")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]<2.35520577 or data["var45"] ==0) and (data["var43"]>=-0.00820540264 and data["var43"] !=0) and (data["var56"]<-1.92028618 or data["var56"] ==0) and (data["var36"]>=1.03473449 and data["var36"] !=0)):
             s.append("19150")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]<2.35520577 or data["var45"] ==0) and (data["var43"]>=-0.00820540264 and data["var43"] !=0) and (data["var56"]>=-1.92028618 and data["var56"] !=0) and (data["var50"]<1.13592231 or data["var50"] ==0)):
             s.append("19151")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]<2.35520577 or data["var45"] ==0) and (data["var43"]>=-0.00820540264 and data["var43"] !=0) and (data["var56"]>=-1.92028618 and data["var56"] !=0) and (data["var50"]>=1.13592231 and data["var50"] !=0)):
             s.append("19152")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]<-1.10171294 or data["var56"] ==0) and (data["var45"]>=2.35520577 and data["var45"] !=0)):
             s.append("1928")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]<1.12533689 or data["var11"] ==0) and (data["var36"]<2.34167767 or data["var36"] ==0) and (data["var08"]<-2.14147997 or data["var08"] ==0) and (data["var58"]<-1.36281943 or data["var58"] ==0)):
             s.append("19153")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]<1.12533689 or data["var11"] ==0) and (data["var36"]<2.34167767 or data["var36"] ==0) and (data["var08"]<-2.14147997 or data["var08"] ==0) and (data["var58"]>=-1.36281943 and data["var58"] !=0)):
             s.append("19154")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]<1.12533689 or data["var11"] ==0) and (data["var36"]<2.34167767 or data["var36"] ==0) and (data["var08"]>=-2.14147997 and data["var08"] !=0) and (data["var56"]<1.15207481 or data["var56"] ==0)):
             s.append("19155")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]<1.12533689 or data["var11"] ==0) and (data["var36"]<2.34167767 or data["var36"] ==0) and (data["var08"]>=-2.14147997 and data["var08"] !=0) and (data["var56"]>=1.15207481 and data["var56"] !=0)):
             s.append("19156")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]<1.12533689 or data["var11"] ==0) and (data["var36"]>=2.34167767 and data["var36"] !=0) and (data["var40"]<-1.07793784 or data["var40"] ==0)):
             s.append("1995")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]<1.12533689 or data["var11"] ==0) and (data["var36"]>=2.34167767 and data["var36"] !=0) and (data["var40"]>=-1.07793784 and data["var40"] !=0)):
             s.append("1996")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]>=1.12533689 and data["var11"] !=0) and (data["var24"]<-1.19690299 or data["var24"] ==0) and (data["var60"]<-1.5970782 or data["var60"] ==0)):
             s.append("1997")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]>=1.12533689 and data["var11"] !=0) and (data["var24"]<-1.19690299 or data["var24"] ==0) and (data["var60"]>=-1.5970782 and data["var60"] !=0)):
             s.append("1998")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]>=1.12533689 and data["var11"] !=0) and (data["var24"]>=-1.19690299 and data["var24"] !=0) and (data["var19"]<0.361657649 or data["var19"] ==0) and (data["var13"]<-2.20922446 or data["var13"] ==0)):
             s.append("19157")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]>=1.12533689 and data["var11"] !=0) and (data["var24"]>=-1.19690299 and data["var24"] !=0) and (data["var19"]<0.361657649 or data["var19"] ==0) and (data["var13"]>=-2.20922446 and data["var13"] !=0)):
             s.append("19158")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]>=1.12533689 and data["var11"] !=0) and (data["var24"]>=-1.19690299 and data["var24"] !=0) and (data["var19"]>=0.361657649 and data["var19"] !=0) and (data["var46"]<1.74122214 or data["var46"] ==0)):
             s.append("19159")
    if((data["var03"]>=-2.21224356 and data["var03"] !=0) and (data["var17"]>=1.97562468 and data["var17"] !=0) and (data["var56"]>=-1.10171294 and data["var56"] !=0) and (data["var11"]>=1.12533689 and data["var11"] !=0) and (data["var24"]>=-1.19690299 and data["var24"] !=0) and (data["var19"]>=0.361657649 and data["var19"] !=0) and (data["var46"]>=1.74122214 and data["var46"] !=0)):
             s.append("19160")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]<-0.604852438 or data["var34"] ==0) and (data["var50"]<0.205437258 or data["var50"] ==0) and (data["var46"]<0.986574233 or data["var46"] ==0) and (data["var17"]<1.33653653 or data["var17"] ==0)):
             s.append("2089")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]<-0.604852438 or data["var34"] ==0) and (data["var50"]<0.205437258 or data["var50"] ==0) and (data["var46"]<0.986574233 or data["var46"] ==0) and (data["var17"]>=1.33653653 and data["var17"] !=0)):
             s.append("2090")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]<-0.604852438 or data["var34"] ==0) and (data["var50"]<0.205437258 or data["var50"] ==0) and (data["var46"]>=0.986574233 and data["var46"] !=0) and (data["var14"]<-0.216020837 or data["var14"] ==0)):
             s.append("2091")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]<-0.604852438 or data["var34"] ==0) and (data["var50"]<0.205437258 or data["var50"] ==0) and (data["var46"]>=0.986574233 and data["var46"] !=0) and (data["var14"]>=-0.216020837 and data["var14"] !=0)):
             s.append("2092")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]<-0.604852438 or data["var34"] ==0) and (data["var50"]>=0.205437258 and data["var50"] !=0) and (data["var19"]<-1.18571663 or data["var19"] ==0) and (data["var22"]<-2.64883232 or data["var22"] ==0)):
             s.append("2093")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]<-0.604852438 or data["var34"] ==0) and (data["var50"]>=0.205437258 and data["var50"] !=0) and (data["var19"]<-1.18571663 or data["var19"] ==0) and (data["var22"]>=-2.64883232 and data["var22"] !=0)):
             s.append("2094")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]<-0.604852438 or data["var34"] ==0) and (data["var50"]>=0.205437258 and data["var50"] !=0) and (data["var19"]>=-1.18571663 and data["var19"] !=0)):
             s.append("2056")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]<-0.817772865 or data["var41"] ==0) and (data["var23"]<-1.31559992 or data["var23"] ==0) and (data["var29"]<-0.664951146 or data["var29"] ==0)):
             s.append("2095")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]<-0.817772865 or data["var41"] ==0) and (data["var23"]<-1.31559992 or data["var23"] ==0) and (data["var29"]>=-0.664951146 and data["var29"] !=0)):
             s.append("2096")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]<-0.817772865 or data["var41"] ==0) and (data["var23"]>=-1.31559992 and data["var23"] !=0) and (data["var49"]<-0.0832453668 or data["var49"] ==0)):
             s.append("2097")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]<-0.817772865 or data["var41"] ==0) and (data["var23"]>=-1.31559992 and data["var23"] !=0) and (data["var49"]>=-0.0832453668 and data["var49"] !=0)):
             s.append("2098")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]>=-0.817772865 and data["var41"] !=0) and (data["var56"]<0.996778846 or data["var56"] ==0) and (data["var02"]<-1.24276614 or data["var02"] ==0)):
             s.append("2099")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]>=-0.817772865 and data["var41"] !=0) and (data["var56"]<0.996778846 or data["var56"] ==0) and (data["var02"]>=-1.24276614 and data["var02"] !=0)):
             s.append("20100")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]>=-0.817772865 and data["var41"] !=0) and (data["var56"]>=0.996778846 and data["var56"] !=0) and (data["var60"]<0.288080752 or data["var60"] ==0)):
             s.append("20101")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]<-2.34297276 or data["var22"] ==0) and (data["var34"]>=-0.604852438 and data["var34"] !=0) and (data["var41"]>=-0.817772865 and data["var41"] !=0) and (data["var56"]>=0.996778846 and data["var56"] !=0) and (data["var60"]>=0.288080752 and data["var60"] !=0)):
             s.append("20102")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]<1.79588187 or data["var55"] ==0) and (data["var50"]<-2.49650669 or data["var50"] ==0) and (data["var12"]<0.264621735 or data["var12"] ==0)):
             s.append("20103")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]<1.79588187 or data["var55"] ==0) and (data["var50"]<-2.49650669 or data["var50"] ==0) and (data["var12"]>=0.264621735 and data["var12"] !=0)):
             s.append("20104")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]<1.79588187 or data["var55"] ==0) and (data["var50"]>=-2.49650669 and data["var50"] !=0) and (data["var13"]<2.21466827 or data["var13"] ==0)):
             s.append("20105")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]<1.79588187 or data["var55"] ==0) and (data["var50"]>=-2.49650669 and data["var50"] !=0) and (data["var13"]>=2.21466827 and data["var13"] !=0)):
             s.append("20106")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]>=1.79588187 and data["var55"] !=0) and (data["var56"]<0.618611276 or data["var56"] ==0) and (data["var12"]<2.45080519 or data["var12"] ==0)):
             s.append("20107")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]>=1.79588187 and data["var55"] !=0) and (data["var56"]<0.618611276 or data["var56"] ==0) and (data["var12"]>=2.45080519 and data["var12"] !=0)):
             s.append("20108")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]>=1.79588187 and data["var55"] !=0) and (data["var56"]>=0.618611276 and data["var56"] !=0) and (data["var18"]<0.2726385 or data["var18"] ==0)):
             s.append("20109")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]<-1.71042538 or data["var06"] ==0) and (data["var55"]>=1.79588187 and data["var55"] !=0) and (data["var56"]>=0.618611276 and data["var56"] !=0) and (data["var18"]>=0.2726385 and data["var18"] !=0)):
             s.append("20110")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]<-1.88194823 or data["var02"] ==0) and (data["var42"]<-2.3759563 or data["var42"] ==0) and (data["var01"]<-1.5585475 or data["var01"] ==0)):
             s.append("20111")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]<-1.88194823 or data["var02"] ==0) and (data["var42"]<-2.3759563 or data["var42"] ==0) and (data["var01"]>=-1.5585475 and data["var01"] !=0)):
             s.append("20112")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]<-1.88194823 or data["var02"] ==0) and (data["var42"]>=-2.3759563 and data["var42"] !=0) and (data["var40"]<-1.69417191 or data["var40"] ==0)):
             s.append("20113")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]<-1.88194823 or data["var02"] ==0) and (data["var42"]>=-2.3759563 and data["var42"] !=0) and (data["var40"]>=-1.69417191 and data["var40"] !=0)):
             s.append("20114")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]>=-1.88194823 and data["var02"] !=0) and (data["var26"]<2.20655394 or data["var26"] ==0) and (data["var08"]<2.42754722 or data["var08"] ==0)):
             s.append("20115")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]>=-1.88194823 and data["var02"] !=0) and (data["var26"]<2.20655394 or data["var26"] ==0) and (data["var08"]>=2.42754722 and data["var08"] !=0)):
             s.append("20116")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]>=-1.88194823 and data["var02"] !=0) and (data["var26"]>=2.20655394 and data["var26"] !=0) and (data["var12"]<1.1826818 or data["var12"] ==0)):
             s.append("20117")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]<2.00610709 or data["var14"] ==0) and (data["var22"]>=-2.34297276 and data["var22"] !=0) and (data["var06"]>=-1.71042538 and data["var06"] !=0) and (data["var02"]>=-1.88194823 and data["var02"] !=0) and (data["var26"]>=2.20655394 and data["var26"] !=0) and (data["var12"]>=1.1826818 and data["var12"] !=0)):
             s.append("20118")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]<-1.86734295 or data["var40"] ==0) and (data["var12"]<1.54831624 or data["var12"] ==0)):
             s.append("2039")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]<-1.86734295 or data["var40"] ==0) and (data["var12"]>=1.54831624 and data["var12"] !=0)):
             s.append("2040")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]<3.17770839 or data["var14"] ==0) and (data["var06"]<1.00731945 or data["var06"] ==0) and (data["var26"]<-0.0797978342 or data["var26"] ==0)):
             s.append("20119")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]<3.17770839 or data["var14"] ==0) and (data["var06"]<1.00731945 or data["var06"] ==0) and (data["var26"]>=-0.0797978342 and data["var26"] !=0)):
             s.append("20120")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]<3.17770839 or data["var14"] ==0) and (data["var06"]>=1.00731945 and data["var06"] !=0) and (data["var19"]<-1.66247904 or data["var19"] ==0)):
             s.append("20121")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]<3.17770839 or data["var14"] ==0) and (data["var06"]>=1.00731945 and data["var06"] !=0) and (data["var19"]>=-1.66247904 and data["var19"] !=0)):
             s.append("20122")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]>=3.17770839 and data["var14"] !=0) and (data["var30"]<-0.403168857 or data["var30"] ==0) and (data["var07"]<1.20233464 or data["var07"] ==0)):
             s.append("20123")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]>=3.17770839 and data["var14"] !=0) and (data["var30"]<-0.403168857 or data["var30"] ==0) and (data["var07"]>=1.20233464 and data["var07"] !=0)):
             s.append("20124")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]>=3.17770839 and data["var14"] !=0) and (data["var30"]>=-0.403168857 and data["var30"] !=0) and (data["var30"]<1.28291261 or data["var30"] ==0)):
             s.append("20125")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]<-0.531233609 or data["var55"] ==0) and (data["var40"]>=-1.86734295 and data["var40"] !=0) and (data["var14"]>=3.17770839 and data["var14"] !=0) and (data["var30"]>=-0.403168857 and data["var30"] !=0) and (data["var30"]>=1.28291261 and data["var30"] !=0)):
             s.append("20126")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]<-1.11906862 or data["var01"] ==0) and (data["var04"]<-0.0213355944 or data["var04"] ==0) and (data["var26"]<1.1607002 or data["var26"] ==0)):
             s.append("20127")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]<-1.11906862 or data["var01"] ==0) and (data["var04"]<-0.0213355944 or data["var04"] ==0) and (data["var26"]>=1.1607002 and data["var26"] !=0)):
             s.append("20128")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]<-1.11906862 or data["var01"] ==0) and (data["var04"]>=-0.0213355944 and data["var04"] !=0) and (data["var18"]<1.38316596 or data["var18"] ==0)):
             s.append("20129")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]<-1.11906862 or data["var01"] ==0) and (data["var04"]>=-0.0213355944 and data["var04"] !=0) and (data["var18"]>=1.38316596 and data["var18"] !=0)):
             s.append("20130")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]>=-1.11906862 and data["var01"] !=0) and (data["var08"]<-0.50338465 or data["var08"] ==0) and (data["var58"]<-0.0419892408 or data["var58"] ==0)):
             s.append("20131")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]>=-1.11906862 and data["var01"] !=0) and (data["var08"]<-0.50338465 or data["var08"] ==0) and (data["var58"]>=-0.0419892408 and data["var58"] !=0)):
             s.append("20132")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]>=-1.11906862 and data["var01"] !=0) and (data["var08"]>=-0.50338465 and data["var08"] !=0) and (data["var55"]<2.05947781 or data["var55"] ==0)):
             s.append("20133")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]<1.08309054 or data["var12"] ==0) and (data["var01"]>=-1.11906862 and data["var01"] !=0) and (data["var08"]>=-0.50338465 and data["var08"] !=0) and (data["var55"]>=2.05947781 and data["var55"] !=0)):
             s.append("20134")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]>=1.08309054 and data["var12"] !=0) and (data["var22"]<0.324987382 or data["var22"] ==0) and (data["var60"]<-0.0342273004 or data["var60"] ==0) and (data["var50"]<0.920419574 or data["var50"] ==0)):
             s.append("20135")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]>=1.08309054 and data["var12"] !=0) and (data["var22"]<0.324987382 or data["var22"] ==0) and (data["var60"]<-0.0342273004 or data["var60"] ==0) and (data["var50"]>=0.920419574 and data["var50"] !=0)):
             s.append("20136")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]>=1.08309054 and data["var12"] !=0) and (data["var22"]<0.324987382 or data["var22"] ==0) and (data["var60"]>=-0.0342273004 and data["var60"] !=0) and (data["var36"]<-0.997987866 or data["var36"] ==0)):
             s.append("20137")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]>=1.08309054 and data["var12"] !=0) and (data["var22"]<0.324987382 or data["var22"] ==0) and (data["var60"]>=-0.0342273004 and data["var60"] !=0) and (data["var36"]>=-0.997987866 and data["var36"] !=0)):
             s.append("20138")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]>=1.08309054 and data["var12"] !=0) and (data["var22"]>=0.324987382 and data["var22"] !=0) and (data["var20"]<1.40524673 or data["var20"] ==0) and (data["var49"]<1.39597535 or data["var49"] ==0)):
             s.append("20139")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]>=1.08309054 and data["var12"] !=0) and (data["var22"]>=0.324987382 and data["var22"] !=0) and (data["var20"]<1.40524673 or data["var20"] ==0) and (data["var49"]>=1.39597535 and data["var49"] !=0)):
             s.append("20140")
    if((data["var39"]<2.87884068 or data["var39"] ==0) and (data["var14"]>=2.00610709 and data["var14"] !=0) and (data["var55"]>=-0.531233609 and data["var55"] !=0) and (data["var12"]>=1.08309054 and data["var12"] !=0) and (data["var22"]>=0.324987382 and data["var22"] !=0) and (data["var20"]>=1.40524673 and data["var20"] !=0)):
             s.append("2080")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]<0.741672516 or data["var29"] ==0) and (data["var15"]<-0.428626657 or data["var15"] ==0) and (data["var31"]<-1.39683819 or data["var31"] ==0)):
             s.append("2047")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]<0.741672516 or data["var29"] ==0) and (data["var15"]<-0.428626657 or data["var15"] ==0) and (data["var31"]>=-1.39683819 and data["var31"] !=0)):
             s.append("2048")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]<0.741672516 or data["var29"] ==0) and (data["var15"]>=-0.428626657 and data["var15"] !=0) and (data["var07"]<-0.772153735 or data["var07"] ==0) and (data["var54"]<0.461566597 or data["var54"] ==0)):
             s.append("2081")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]<0.741672516 or data["var29"] ==0) and (data["var15"]>=-0.428626657 and data["var15"] !=0) and (data["var07"]<-0.772153735 or data["var07"] ==0) and (data["var54"]>=0.461566597 and data["var54"] !=0)):
             s.append("2082")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]<0.741672516 or data["var29"] ==0) and (data["var15"]>=-0.428626657 and data["var15"] !=0) and (data["var07"]>=-0.772153735 and data["var07"] !=0) and (data["var05"]<-1.75475085 or data["var05"] ==0)):
             s.append("2083")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]<0.741672516 or data["var29"] ==0) and (data["var15"]>=-0.428626657 and data["var15"] !=0) and (data["var07"]>=-0.772153735 and data["var07"] !=0) and (data["var05"]>=-1.75475085 and data["var05"] !=0)):
             s.append("2084")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]>=0.741672516 and data["var29"] !=0) and (data["var49"]<0.864032567 or data["var49"] ==0)):
             s.append("2025")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]<-0.352170169 or data["var44"] ==0) and (data["var29"]>=0.741672516 and data["var29"] !=0) and (data["var49"]>=0.864032567 and data["var49"] !=0)):
             s.append("2026")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]<0.0572583079 or data["var20"] ==0) and (data["var39"]<3.30530715 or data["var39"] ==0) and (data["var03"]<0.348291337 or data["var03"] ==0) and (data["var47"]<0.841072917 or data["var47"] ==0)):
             s.append("20141")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]<0.0572583079 or data["var20"] ==0) and (data["var39"]<3.30530715 or data["var39"] ==0) and (data["var03"]<0.348291337 or data["var03"] ==0) and (data["var47"]>=0.841072917 and data["var47"] !=0)):
             s.append("20142")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]<0.0572583079 or data["var20"] ==0) and (data["var39"]<3.30530715 or data["var39"] ==0) and (data["var03"]>=0.348291337 and data["var03"] !=0) and (data["var33"]<-0.513513327 or data["var33"] ==0)):
             s.append("20143")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]<0.0572583079 or data["var20"] ==0) and (data["var39"]<3.30530715 or data["var39"] ==0) and (data["var03"]>=0.348291337 and data["var03"] !=0) and (data["var33"]>=-0.513513327 and data["var33"] !=0)):
             s.append("20144")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]<0.0572583079 or data["var20"] ==0) and (data["var39"]>=3.30530715 and data["var39"] !=0) and (data["var27"]<1.43531621 or data["var27"] ==0) and (data["var19"]<-1.42374289 or data["var19"] ==0)):
             s.append("20145")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]<0.0572583079 or data["var20"] ==0) and (data["var39"]>=3.30530715 and data["var39"] !=0) and (data["var27"]<1.43531621 or data["var27"] ==0) and (data["var19"]>=-1.42374289 and data["var19"] !=0)):
             s.append("20146")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]<0.0572583079 or data["var20"] ==0) and (data["var39"]>=3.30530715 and data["var39"] !=0) and (data["var27"]>=1.43531621 and data["var27"] !=0)):
             s.append("2088")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]<3.98970175 or data["var39"] ==0) and (data["var20"]>=0.0572583079 and data["var20"] !=0)):
             s.append("2028")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]>=3.98970175 and data["var39"] !=0) and (data["var58"]<-0.465941787 or data["var58"] ==0)):
             s.append("2029")
    if((data["var39"]>=2.87884068 and data["var39"] !=0) and (data["var44"]>=-0.352170169 and data["var44"] !=0) and (data["var39"]>=3.98970175 and data["var39"] !=0) and (data["var58"]>=-0.465941787 and data["var58"] !=0)):
             s.append("2030")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]<-1.48645139 or data["var10"] ==0) and (data["var34"]<0.915339231 or data["var34"] ==0) and (data["var02"]<1.57522154 or data["var02"] ==0) and (data["var14"]<1.92258811 or data["var14"] ==0)):
             s.append("2157")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]<-1.48645139 or data["var10"] ==0) and (data["var34"]<0.915339231 or data["var34"] ==0) and (data["var02"]<1.57522154 or data["var02"] ==0) and (data["var14"]>=1.92258811 and data["var14"] !=0)):
             s.append("2158")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]<-1.48645139 or data["var10"] ==0) and (data["var34"]<0.915339231 or data["var34"] ==0) and (data["var02"]>=1.57522154 and data["var02"] !=0)):
             s.append("2132")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]<-1.48645139 or data["var10"] ==0) and (data["var34"]>=0.915339231 and data["var34"] !=0) and (data["var51"]<-0.150910825 or data["var51"] ==0)):
             s.append("2133")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]<-1.48645139 or data["var10"] ==0) and (data["var34"]>=0.915339231 and data["var34"] !=0) and (data["var51"]>=-0.150910825 and data["var51"] !=0)):
             s.append("2134")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]<0.347085327 or data["var46"] ==0) and (data["var31"]<1.25099039 or data["var31"] ==0) and (data["var03"]<1.92995501 or data["var03"] ==0) and (data["var13"]<-1.46912599 or data["var13"] ==0)):
             s.append("2193")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]<0.347085327 or data["var46"] ==0) and (data["var31"]<1.25099039 or data["var31"] ==0) and (data["var03"]<1.92995501 or data["var03"] ==0) and (data["var13"]>=-1.46912599 and data["var13"] !=0)):
             s.append("2194")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]<0.347085327 or data["var46"] ==0) and (data["var31"]<1.25099039 or data["var31"] ==0) and (data["var03"]>=1.92995501 and data["var03"] !=0)):
             s.append("2160")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]<0.347085327 or data["var46"] ==0) and (data["var31"]>=1.25099039 and data["var31"] !=0) and (data["var05"]<0.897939801 or data["var05"] ==0) and (data["var24"]<0.100787759 or data["var24"] ==0)):
             s.append("2195")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]<0.347085327 or data["var46"] ==0) and (data["var31"]>=1.25099039 and data["var31"] !=0) and (data["var05"]<0.897939801 or data["var05"] ==0) and (data["var24"]>=0.100787759 and data["var24"] !=0)):
             s.append("2196")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]<0.347085327 or data["var46"] ==0) and (data["var31"]>=1.25099039 and data["var31"] !=0) and (data["var05"]>=0.897939801 and data["var05"] !=0) and (data["var31"]<1.45736444 or data["var31"] ==0)):
             s.append("2197")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]<0.347085327 or data["var46"] ==0) and (data["var31"]>=1.25099039 and data["var31"] !=0) and (data["var05"]>=0.897939801 and data["var05"] !=0) and (data["var31"]>=1.45736444 and data["var31"] !=0)):
             s.append("2198")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]>=0.347085327 and data["var46"] !=0) and (data["var47"]<0.155422747 or data["var47"] ==0) and (data["var44"]<-1.11119437 or data["var44"] ==0)):
             s.append("2163")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]>=0.347085327 and data["var46"] !=0) and (data["var47"]<0.155422747 or data["var47"] ==0) and (data["var44"]>=-1.11119437 and data["var44"] !=0) and (data["var09"]<-0.315912366 or data["var09"] ==0)):
             s.append("2199")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]>=0.347085327 and data["var46"] !=0) and (data["var47"]<0.155422747 or data["var47"] ==0) and (data["var44"]>=-1.11119437 and data["var44"] !=0) and (data["var09"]>=-0.315912366 and data["var09"] !=0)):
             s.append("21100")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]>=0.347085327 and data["var46"] !=0) and (data["var47"]>=0.155422747 and data["var47"] !=0) and (data["var48"]<-1.39839149 or data["var48"] ==0)):
             s.append("2165")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]>=0.347085327 and data["var46"] !=0) and (data["var47"]>=0.155422747 and data["var47"] !=0) and (data["var48"]>=-1.39839149 and data["var48"] !=0) and (data["var38"]<2.04285049 or data["var38"] ==0)):
             s.append("21101")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]<-2.25628233 or data["var15"] ==0) and (data["var10"]>=-1.48645139 and data["var10"] !=0) and (data["var46"]>=0.347085327 and data["var46"] !=0) and (data["var47"]>=0.155422747 and data["var47"] !=0) and (data["var48"]>=-1.39839149 and data["var48"] !=0) and (data["var38"]>=2.04285049 and data["var38"] !=0)):
             s.append("21102")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]<-1.43742621 or data["var04"] ==0) and (data["var03"]<1.63717651 or data["var03"] ==0) and (data["var57"]<1.87208569 or data["var57"] ==0) and (data["var45"]<1.72389889 or data["var45"] ==0)):
             s.append("21103")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]<-1.43742621 or data["var04"] ==0) and (data["var03"]<1.63717651 or data["var03"] ==0) and (data["var57"]<1.87208569 or data["var57"] ==0) and (data["var45"]>=1.72389889 and data["var45"] !=0)):
             s.append("21104")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]<-1.43742621 or data["var04"] ==0) and (data["var03"]<1.63717651 or data["var03"] ==0) and (data["var57"]>=1.87208569 and data["var57"] !=0)):
             s.append("2168")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]<-1.43742621 or data["var04"] ==0) and (data["var03"]>=1.63717651 and data["var03"] !=0) and (data["var25"]<-0.255693227 or data["var25"] ==0)):
             s.append("2169")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]<-1.43742621 or data["var04"] ==0) and (data["var03"]>=1.63717651 and data["var03"] !=0) and (data["var25"]>=-0.255693227 and data["var25"] !=0)):
             s.append("2170")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]<-0.311838895 or data["var20"] ==0) and (data["var60"]<0.979001999 or data["var60"] ==0) and (data["var41"]<0.24972555 or data["var41"] ==0)):
             s.append("21105")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]<-0.311838895 or data["var20"] ==0) and (data["var60"]<0.979001999 or data["var60"] ==0) and (data["var41"]>=0.24972555 and data["var41"] !=0)):
             s.append("21106")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]<-0.311838895 or data["var20"] ==0) and (data["var60"]>=0.979001999 and data["var60"] !=0) and (data["var36"]<0.328881204 or data["var36"] ==0)):
             s.append("21107")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]<-0.311838895 or data["var20"] ==0) and (data["var60"]>=0.979001999 and data["var60"] !=0) and (data["var36"]>=0.328881204 and data["var36"] !=0)):
             s.append("21108")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]>=-0.311838895 and data["var20"] !=0) and (data["var04"]<-0.828349948 or data["var04"] ==0) and (data["var56"]<1.20171356 or data["var56"] ==0)):
             s.append("21109")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]>=-0.311838895 and data["var20"] !=0) and (data["var04"]<-0.828349948 or data["var04"] ==0) and (data["var56"]>=1.20171356 and data["var56"] !=0)):
             s.append("21110")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]>=-0.311838895 and data["var20"] !=0) and (data["var04"]>=-0.828349948 and data["var04"] !=0) and (data["var25"]<-0.947934151 or data["var25"] ==0)):
             s.append("21111")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]<-2.15547037 or data["var11"] ==0) and (data["var04"]>=-1.43742621 and data["var04"] !=0) and (data["var20"]>=-0.311838895 and data["var20"] !=0) and (data["var04"]>=-0.828349948 and data["var04"] !=0) and (data["var25"]>=-0.947934151 and data["var25"] !=0)):
             s.append("21112")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]<1.42474592 or data["var08"] ==0) and (data["var10"]<1.55613995 or data["var10"] ==0) and (data["var02"]<-0.528532147 or data["var02"] ==0)):
             s.append("21113")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]<1.42474592 or data["var08"] ==0) and (data["var10"]<1.55613995 or data["var10"] ==0) and (data["var02"]>=-0.528532147 and data["var02"] !=0)):
             s.append("21114")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]<1.42474592 or data["var08"] ==0) and (data["var10"]>=1.55613995 and data["var10"] !=0) and (data["var46"]<1.30475402 or data["var46"] ==0)):
             s.append("21115")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]<1.42474592 or data["var08"] ==0) and (data["var10"]>=1.55613995 and data["var10"] !=0) and (data["var46"]>=1.30475402 and data["var46"] !=0)):
             s.append("21116")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]>=1.42474592 and data["var08"] !=0) and (data["var60"]<-1.40035057 or data["var60"] ==0) and (data["var58"]<0.962619185 or data["var58"] ==0)):
             s.append("21117")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]>=1.42474592 and data["var08"] !=0) and (data["var60"]<-1.40035057 or data["var60"] ==0) and (data["var58"]>=0.962619185 and data["var58"] !=0)):
             s.append("21118")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]>=1.42474592 and data["var08"] !=0) and (data["var60"]>=-1.40035057 and data["var60"] !=0) and (data["var22"]<-0.230301559 or data["var22"] ==0)):
             s.append("21119")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]<-2.42015219 or data["var09"] ==0) and (data["var08"]>=1.42474592 and data["var08"] !=0) and (data["var60"]>=-1.40035057 and data["var60"] !=0) and (data["var22"]>=-0.230301559 and data["var22"] !=0)):
             s.append("21120")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]<1.77034044 or data["var09"] ==0) and (data["var13"]<2.24492478 or data["var13"] ==0) and (data["var59"]<-2.42695498 or data["var59"] ==0)):
             s.append("21121")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]<1.77034044 or data["var09"] ==0) and (data["var13"]<2.24492478 or data["var13"] ==0) and (data["var59"]>=-2.42695498 and data["var59"] !=0)):
             s.append("21122")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]<1.77034044 or data["var09"] ==0) and (data["var13"]>=2.24492478 and data["var13"] !=0) and (data["var37"]<1.53947175 or data["var37"] ==0)):
             s.append("21123")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]<1.77034044 or data["var09"] ==0) and (data["var13"]>=2.24492478 and data["var13"] !=0) and (data["var37"]>=1.53947175 and data["var37"] !=0)):
             s.append("21124")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]>=1.77034044 and data["var09"] !=0) and (data["var32"]<1.35283935 or data["var32"] ==0) and (data["var46"]<-1.49424577 or data["var46"] ==0)):
             s.append("21125")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]>=1.77034044 and data["var09"] !=0) and (data["var32"]<1.35283935 or data["var32"] ==0) and (data["var46"]>=-1.49424577 and data["var46"] !=0)):
             s.append("21126")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]>=1.77034044 and data["var09"] !=0) and (data["var32"]>=1.35283935 and data["var32"] !=0) and (data["var28"]<-0.721589088 or data["var28"] ==0)):
             s.append("21127")
    if((data["var53"]<2.035954 or data["var53"] ==0) and (data["var15"]>=-2.25628233 and data["var15"] !=0) and (data["var11"]>=-2.15547037 and data["var11"] !=0) and (data["var09"]>=-2.42015219 and data["var09"] !=0) and (data["var09"]>=1.77034044 and data["var09"] !=0) and (data["var32"]>=1.35283935 and data["var32"] !=0) and (data["var28"]>=-0.721589088 and data["var28"] !=0)):
             s.append("21128")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]<-2.41044641 or data["var54"] ==0) and (data["var32"]<1.34891415 or data["var32"] ==0)):
             s.append("2147")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]<-2.41044641 or data["var54"] ==0) and (data["var32"]>=1.34891415 and data["var32"] !=0)):
             s.append("2148")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]<-1.68892026 or data["var27"] ==0) and (data["var15"]<-0.73715657 or data["var15"] ==0) and (data["var18"]<-1.07246852 or data["var18"] ==0)):
             s.append("21129")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]<-1.68892026 or data["var27"] ==0) and (data["var15"]<-0.73715657 or data["var15"] ==0) and (data["var18"]>=-1.07246852 and data["var18"] !=0)):
             s.append("21130")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]<-1.68892026 or data["var27"] ==0) and (data["var15"]>=-0.73715657 and data["var15"] !=0) and (data["var30"]<1.18068039 or data["var30"] ==0)):
             s.append("21131")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]<-1.68892026 or data["var27"] ==0) and (data["var15"]>=-0.73715657 and data["var15"] !=0) and (data["var30"]>=1.18068039 and data["var30"] !=0)):
             s.append("21132")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]>=-1.68892026 and data["var27"] !=0) and (data["var56"]<-1.85342574 or data["var56"] ==0) and (data["var25"]<0.175923139 or data["var25"] ==0)):
             s.append("21133")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]>=-1.68892026 and data["var27"] !=0) and (data["var56"]<-1.85342574 or data["var56"] ==0) and (data["var25"]>=0.175923139 and data["var25"] !=0)):
             s.append("21134")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]>=-1.68892026 and data["var27"] !=0) and (data["var56"]>=-1.85342574 and data["var56"] !=0) and (data["var12"]<-2.51065207 or data["var12"] ==0)):
             s.append("21135")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]<1.20311689 or data["var25"] ==0) and (data["var54"]>=-2.41044641 and data["var54"] !=0) and (data["var27"]>=-1.68892026 and data["var27"] !=0) and (data["var56"]>=-1.85342574 and data["var56"] !=0) and (data["var12"]>=-2.51065207 and data["var12"] !=0)):
             s.append("21136")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]<-0.677216589 or data["var02"] ==0) and (data["var45"]<-1.88479376 or data["var45"] ==0)):
             s.append("2151")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]<-0.677216589 or data["var02"] ==0) and (data["var45"]>=-1.88479376 and data["var45"] !=0) and (data["var07"]<1.83146381 or data["var07"] ==0) and (data["var19"]<-1.16069865 or data["var19"] ==0)):
             s.append("21137")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]<-0.677216589 or data["var02"] ==0) and (data["var45"]>=-1.88479376 and data["var45"] !=0) and (data["var07"]<1.83146381 or data["var07"] ==0) and (data["var19"]>=-1.16069865 and data["var19"] !=0)):
             s.append("21138")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]<-0.677216589 or data["var02"] ==0) and (data["var45"]>=-1.88479376 and data["var45"] !=0) and (data["var07"]>=1.83146381 and data["var07"] !=0)):
             s.append("2188")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]>=-0.677216589 and data["var02"] !=0) and (data["var43"]<1.18091738 or data["var43"] ==0) and (data["var02"]<-0.418385029 or data["var02"] ==0)):
             s.append("2189")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]>=-0.677216589 and data["var02"] !=0) and (data["var43"]<1.18091738 or data["var43"] ==0) and (data["var02"]>=-0.418385029 and data["var02"] !=0) and (data["var54"]<-1.9488914 or data["var54"] ==0)):
             s.append("21139")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]>=-0.677216589 and data["var02"] !=0) and (data["var43"]<1.18091738 or data["var43"] ==0) and (data["var02"]>=-0.418385029 and data["var02"] !=0) and (data["var54"]>=-1.9488914 and data["var54"] !=0)):
             s.append("21140")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]>=-0.677216589 and data["var02"] !=0) and (data["var43"]>=1.18091738 and data["var43"] !=0) and (data["var11"]<0.138685539 or data["var11"] ==0)):
             s.append("2191")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]>=-0.677216589 and data["var02"] !=0) and (data["var43"]>=1.18091738 and data["var43"] !=0) and (data["var11"]>=0.138685539 and data["var11"] !=0) and (data["var01"]<-0.370640814 or data["var01"] ==0)):
             s.append("21141")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]<1.82640433 or data["var50"] ==0) and (data["var25"]>=1.20311689 and data["var25"] !=0) and (data["var02"]>=-0.677216589 and data["var02"] !=0) and (data["var43"]>=1.18091738 and data["var43"] !=0) and (data["var11"]>=0.138685539 and data["var11"] !=0) and (data["var01"]>=-0.370640814 and data["var01"] !=0)):
             s.append("21142")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]>=1.82640433 and data["var50"] !=0) and (data["var16"]<-0.75224328 or data["var16"] ==0) and (data["var15"]<-1.13635921 or data["var15"] ==0)):
             s.append("2127")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]>=1.82640433 and data["var50"] !=0) and (data["var16"]<-0.75224328 or data["var16"] ==0) and (data["var15"]>=-1.13635921 and data["var15"] !=0)):
             s.append("2128")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]>=1.82640433 and data["var50"] !=0) and (data["var16"]>=-0.75224328 and data["var16"] !=0) and (data["var09"]<-2.14409471 or data["var09"] ==0)):
             s.append("2129")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]>=1.82640433 and data["var50"] !=0) and (data["var16"]>=-0.75224328 and data["var16"] !=0) and (data["var09"]>=-2.14409471 and data["var09"] !=0) and (data["var11"]<-1.90817332 or data["var11"] ==0)):
             s.append("2155")
    if((data["var53"]>=2.035954 and data["var53"] !=0) and (data["var50"]>=1.82640433 and data["var50"] !=0) and (data["var16"]>=-0.75224328 and data["var16"] !=0) and (data["var09"]>=-2.14409471 and data["var09"] !=0) and (data["var11"]>=-1.90817332 and data["var11"] !=0)):
             s.append("2156")
    if((data["var29"]<2.01861525 or data["var29"] ==0) and (data["var20"]<-2.02294731 or data["var20"] ==0) and (data["var40"]<-1.72471976 or data["var40"] ==0) and (data["var16"]<-1.08131218 or data["var16"] ==0) and (data["var41"]<0.00941061042 or data["var41"] ==0)):
             s.append("2229")
    if((data["var29"]<2.01861525 or data["var29"] ==0) and (data["var20"]<-2.02294731 or data["var20"] ==0) and (data["var40"]<-1.72471976 or data["var40"] ==0) and (data["var16"]<-1.08131218 or data["var16"] ==0) and (data["var41"]>=0.00941061042 and data["var41"] !=0)):
             s.append("2230")
    if((data["var29"]<2.01861525 or data["var29"] ==0) and (data["var20"]<-2.02294731 or data["var20"] ==0) and (data["var40"]<-1.72471976 or data["var40"] ==0) and (data["var16"]>=-1.08131218 and data["var16"] !=0) and (data["var28"]<-1.65440369 or data["var28"] ==0) and (data["var36"]<-0.61106801 or data["var36"] ==0)):
             s.append("2257")
    return s
