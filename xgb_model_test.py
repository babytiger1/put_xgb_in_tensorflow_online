import numpy as np

def xgb_tree(x, num_booster):
    if num_booster == 0:
        state = 0
        if state == 0:
            state = (1 if x['var03']<-2.21224356 or np.isnan(x['var03'])  else 2)
            if state == 1:
                state = (3 if x['var42']<-0.990586758 or np.isnan(x['var42'])  else 4)
                if state == 3:
                    state = (7 if x['var17']<0.875352502 or np.isnan(x['var17'])  else 8)
                    if state == 7:
                        state = (15 if x['var22']<-2.16016388 or np.isnan(x['var22'])  else 16)
                        if state == 15:
                            return "015"
                        if state == 16:
                            state = (31 if x['var40']<2.91252041 or np.isnan(x['var40'])  else 32)
                            if state == 31:
                                state = (55 if x['var34']<-2.12429523 or np.isnan(x['var34'])  else 56)
                                if state == 55:
                                    return "055"
                                if state == 56:
                                    state = (91 if x['var48']<-2.14176345 or np.isnan(x['var48'])  else 92)
                                    if state == 91:
                                        return "091"
                                    if state == 92:
                                        return "092"
                            if state == 32:
                                return "032"
                    if state == 8:
                        state = (17 if x['var21']<0.520791292 or np.isnan(x['var21'])  else 18)
                        if state == 17:
                            state = (33 if x['var36']<0.256643981 or np.isnan(x['var36'])  else 34)
                            if state == 33:
                                state = (57 if x['var11']<0.986605406 or np.isnan(x['var11'])  else 58)
                                if state == 57:
                                    return "057"
                                if state == 58:
                                    return "058"
                            if state == 34:
                                return "034"
                        if state == 18:
                            return "018"
                if state == 4:
                    state = (9 if x['var30']<1.93426228 or np.isnan(x['var30'])  else 10)
                    if state == 9:
                        state = (19 if x['var21']<-1.24065614 or np.isnan(x['var21'])  else 20)
                        if state == 19:
                            state = (35 if x['var36']<1.22742963 or np.isnan(x['var36'])  else 36)
                            if state == 35:
                                state = (59 if x['var25']<-2.37223125 or np.isnan(x['var25'])  else 60)
                                if state == 59:
                                    return "059"
                                if state == 60:
                                    state = (93 if x['var19']<-2.07533741 or np.isnan(x['var19'])  else 94)
                                    if state == 93:
                                        return "093"
                                    if state == 94:
                                        return "094"
                            if state == 36:
                                state = (61 if x['var27']<0.409669131 or np.isnan(x['var27'])  else 62)
                                if state == 61:
                                    return "061"
                                if state == 62:
                                    return "062"
                        if state == 20:
                            state = (37 if x['var50']<0.734828055 or np.isnan(x['var50'])  else 38)
                            if state == 37:
                                state = (63 if x['var55']<0.436708987 or np.isnan(x['var55'])  else 64)
                                if state == 63:
                                    state = (95 if x['var57']<-1.62303495 or np.isnan(x['var57'])  else 96)
                                    if state == 95:
                                        return "095"
                                    if state == 96:
                                        return "096"
                                if state == 64:
                                    state = (97 if x['var43']<1.17869151 or np.isnan(x['var43'])  else 98)
                                    if state == 97:
                                        return "097"
                                    if state == 98:
                                        return "098"
                            if state == 38:
                                state = (65 if x['var54']<-0.966283083 or np.isnan(x['var54'])  else 66)
                                if state == 65:
                                    state = (99 if x['var37']<-0.452046633 or np.isnan(x['var37'])  else 100)
                                    if state == 99:
                                        return "099"
                                    if state == 100:
                                        return "0100"
                                if state == 66:
                                    state = (101 if x['var57']<0.913910985 or np.isnan(x['var57'])  else 102)
                                    if state == 101:
                                        return "0101"
                                    if state == 102:
                                        return "0102"
                    if state == 10:
                        state = (21 if x['var36']<-2.11946559 or np.isnan(x['var36'])  else 22)
                        if state == 21:
                            return "021"
                        if state == 22:
                            state = (39 if x['var01']<-2.18531823 or np.isnan(x['var01'])  else 40)
                            if state == 39:
                                return "039"
                            if state == 40:
                                return "040"
            if state == 2:
                state = (5 if x['var60']<-2.42058301 or np.isnan(x['var60'])  else 6)
                if state == 5:
                    state = (11 if x['var27']<-0.762669444 or np.isnan(x['var27'])  else 12)
                    if state == 11:
                        state = (23 if x['var13']<2.32689619 or np.isnan(x['var13'])  else 24)
                        if state == 23:
                            state = (41 if x['var40']<2.087677 or np.isnan(x['var40'])  else 42)
                            if state == 41:
                                state = (67 if x['var21']<-2.77495432 or np.isnan(x['var21'])  else 68)
                                if state == 67:
                                    return "067"
                                if state == 68:
                                    state = (103 if x['var28']<2.53138065 or np.isnan(x['var28'])  else 104)
                                    if state == 103:
                                        return "0103"
                                    if state == 104:
                                        return "0104"
                            if state == 42:
                                return "042"
                        if state == 24:
                            return "024"
                    if state == 12:
                        state = (25 if x['var07']<0.349930704 or np.isnan(x['var07'])  else 26)
                        if state == 25:
                            state = (43 if x['var07']<0.00514501799 or np.isnan(x['var07'])  else 44)
                            if state == 43:
                                state = (69 if x['var08']<-0.996021032 or np.isnan(x['var08'])  else 70)
                                if state == 69:
                                    state = (105 if x['var58']<-1.08892322 or np.isnan(x['var58'])  else 106)
                                    if state == 105:
                                        return "0105"
                                    if state == 106:
                                        return "0106"
                                if state == 70:
                                    state = (107 if x['var32']<-0.751752734 or np.isnan(x['var32'])  else 108)
                                    if state == 107:
                                        return "0107"
                                    if state == 108:
                                        return "0108"
                            if state == 44:
                                state = (71 if x['var10']<-1.14291286 or np.isnan(x['var10'])  else 72)
                                if state == 71:
                                    return "071"
                                if state == 72:
                                    return "072"
                        if state == 26:
                            state = (45 if x['var09']<-1.36478424 or np.isnan(x['var09'])  else 46)
                            if state == 45:
                                state = (73 if x['var10']<0.307047158 or np.isnan(x['var10'])  else 74)
                                if state == 73:
                                    return "073"
                                if state == 74:
                                    state = (109 if x['var04']<-0.751992106 or np.isnan(x['var04'])  else 110)
                                    if state == 109:
                                        return "0109"
                                    if state == 110:
                                        return "0110"
                            if state == 46:
                                state = (75 if x['var33']<2.20350504 or np.isnan(x['var33'])  else 76)
                                if state == 75:
                                    state = (111 if x['var49']<1.32830381 or np.isnan(x['var49'])  else 112)
                                    if state == 111:
                                        return "0111"
                                    if state == 112:
                                        return "0112"
                                if state == 76:
                                    state = (113 if x['var12']<0.0812864751 or np.isnan(x['var12'])  else 114)
                                    if state == 113:
                                        return "0113"
                                    if state == 114:
                                        return "0114"
                if state == 6:
                    state = (13 if x['var10']<2.3934474 or np.isnan(x['var10'])  else 14)
                    if state == 13:
                        state = (27 if x['var19']<-1.67283797 or np.isnan(x['var19'])  else 28)
                        if state == 27:
                            state = (47 if x['var19']<-2.92786741 or np.isnan(x['var19'])  else 48)
                            if state == 47:
                                state = (77 if x['var29']<0.335894614 or np.isnan(x['var29'])  else 78)
                                if state == 77:
                                    state = (115 if x['var10']<0.590768337 or np.isnan(x['var10'])  else 116)
                                    if state == 115:
                                        return "0115"
                                    if state == 116:
                                        return "0116"
                                if state == 78:
                                    state = (117 if x['var10']<-1.96327543 or np.isnan(x['var10'])  else 118)
                                    if state == 117:
                                        return "0117"
                                    if state == 118:
                                        return "0118"
                            if state == 48:
                                state = (79 if x['var08']<-2.18796206 or np.isnan(x['var08'])  else 80)
                                if state == 79:
                                    state = (119 if x['var09']<0.103131451 or np.isnan(x['var09'])  else 120)
                                    if state == 119:
                                        return "0119"
                                    if state == 120:
                                        return "0120"
                                if state == 80:
                                    state = (121 if x['var02']<-1.29424691 or np.isnan(x['var02'])  else 122)
                                    if state == 121:
                                        return "0121"
                                    if state == 122:
                                        return "0122"
                        if state == 28:
                            state = (49 if x['var23']<2.38949537 or np.isnan(x['var23'])  else 50)
                            if state == 49:
                                state = (81 if x['var58']<2.4114542 or np.isnan(x['var58'])  else 82)
                                if state == 81:
                                    state = (123 if x['var55']<2.24234605 or np.isnan(x['var55'])  else 124)
                                    if state == 123:
                                        return "0123"
                                    if state == 124:
                                        return "0124"
                                if state == 82:
                                    state = (125 if x['var15']<0.182395443 or np.isnan(x['var15'])  else 126)
                                    if state == 125:
                                        return "0125"
                                    if state == 126:
                                        return "0126"
                            if state == 50:
                                state = (83 if x['var23']<3.4311161 or np.isnan(x['var23'])  else 84)
                                if state == 83:
                                    state = (127 if x['var02']<0.82996726 or np.isnan(x['var02'])  else 128)
                                    if state == 127:
                                        return "0127"
                                    if state == 128:
                                        return "0128"
                                if state == 84:
                                    state = (129 if x['var26']<2.41172004 or np.isnan(x['var26'])  else 130)
                                    if state == 129:
                                        return "0129"
                                    if state == 130:
                                        return "0130"
                    if state == 14:
                        state = (29 if x['var10']<3.56514144 or np.isnan(x['var10'])  else 30)
                        if state == 29:
                            state = (51 if x['var36']<-0.46432656 or np.isnan(x['var36'])  else 52)
                            if state == 51:
                                state = (85 if x['var11']<0.484783292 or np.isnan(x['var11'])  else 86)
                                if state == 85:
                                    state = (131 if x['var48']<0.112738118 or np.isnan(x['var48'])  else 132)
                                    if state == 131:
                                        return "0131"
                                    if state == 132:
                                        return "0132"
                                if state == 86:
                                    state = (133 if x['var29']<-1.0830369 or np.isnan(x['var29'])  else 134)
                                    if state == 133:
                                        return "0133"
                                    if state == 134:
                                        return "0134"
                            if state == 52:
                                state = (87 if x['var31']<1.38491821 or np.isnan(x['var31'])  else 88)
                                if state == 87:
                                    state = (135 if x['var06']<-1.7955395 or np.isnan(x['var06'])  else 136)
                                    if state == 135:
                                        return "0135"
                                    if state == 136:
                                        return "0136"
                                if state == 88:
                                    state = (137 if x['var50']<-0.375242054 or np.isnan(x['var50'])  else 138)
                                    if state == 137:
                                        return "0137"
                                    if state == 138:
                                        return "0138"
                        if state == 30:
                            state = (53 if x['var05']<-1.75495124 or np.isnan(x['var05'])  else 54)
                            if state == 53:
                                return "053"
                            if state == 54:
                                state = (89 if x['var02']<-1.57104671 or np.isnan(x['var02'])  else 90)
                                if state == 89:
                                    return "089"
                                if state == 90:
                                    state = (139 if x['var11']<-1.6163739 or np.isnan(x['var11'])  else 140)
                                    if state == 139:
                                        return "0139"
                                    if state == 140:
                                        return "0140"
