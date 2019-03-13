import pandas as pd
import tensorflow as tf 

def XGBprocess3(data,i):
    if i == 0:
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] < 0.135049999 or data[29] == 0) and (
                data[15] < 47.0349998 or data[15] == 0)):
            return "07"
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] < 0.135049999 or data[29] == 0) and (
                data[15] >= 47.0349998 and data[15] != 0)):
            return "08"
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] >= 0.135049999 and data[29] != 0) and (
                data[3] < 20.2999992 or data[3] == 0) and (data[8] < 0.119749993 or data[8] == 0)):
            return "015"
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] >= 0.135049999 and data[29] != 0) and (
                data[3] < 20.2999992 or data[3] == 0) and (data[8] >= 0.119749993 and data[8] != 0)):
            return "016"
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] >= 0.135049999 and data[29] != 0) and (
                data[3] >= 20.2999992 and data[3] != 0)):
            return "010"
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] < 20.6450005 or data[23] == 0) and (
                data[22] < 17.7399998 or data[22] == 0)):
            return "011"
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] < 20.6450005 or data[23] == 0) and (
                data[22] >= 17.7399998 and data[22] != 0)):
            return "012"
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] < 0.0488649979 or data[9] == 0) and (data[17] < 0.0171750002 or data[17] == 0)):
            return "017"
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] < 0.0488649979 or data[9] == 0) and (data[17] >= 0.0171750002 and data[17] != 0)):
            return "018"
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] >= 0.0488649979 and data[9] != 0) and (data[6] < 0.0827400014 or data[6] == 0)):
            return "019"
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] >= 0.0488649979 and data[9] != 0) and (data[6] >= 0.0827400014 and data[6] != 0)):
            return "020"
    if i == 1:
        if ((data[29] < 0.150999993 or data[29] == 0) and (data[25] < 957.450012 or data[25] == 0) and (
                data[3] < 21.4349995 or data[3] == 0) and (data[13] < 2.09749985 or data[13] == 0)):
            return "111"
        if ((data[29] < 0.150999993 or data[29] == 0) and (data[25] < 957.450012 or data[25] == 0) and (
                data[3] < 21.4349995 or data[3] == 0) and (data[13] >= 2.09749985 and data[13] != 0)):
            return "112"
        if ((data[29] < 0.150999993 or data[29] == 0) and (data[25] < 957.450012 or data[25] == 0) and (
                data[3] >= 21.4349995 and data[3] != 0) and (data[22] < 14.4300003 or data[22] == 0)):
            return "113"
        if ((data[29] < 0.150999993 or data[29] == 0) and (data[25] < 957.450012 or data[25] == 0) and (
                data[3] >= 21.4349995 and data[3] != 0) and (data[22] >= 14.4300003 and data[22] != 0) and (
                data[30] < 0.269600004 or data[30] == 0)):
            return "115"
        if ((data[29] < 0.150999993 or data[29] == 0) and (data[25] < 957.450012 or data[25] == 0) and (
                data[3] >= 21.4349995 and data[3] != 0) and (data[22] >= 14.4300003 and data[22] != 0) and (
                data[30] >= 0.269600004 and data[30] != 0)):
            return "116"
        if ((data[29] < 0.150999993 or data[29] == 0) and (data[25] >= 957.450012 and data[25] != 0) and (
                data[28] < 0.222950011 or data[28] == 0)):
            return "19"
        if ((data[29] < 0.150999993 or data[29] == 0) and (data[25] >= 957.450012 and data[25] != 0) and (
                data[28] >= 0.222950011 and data[28] != 0)):
            return "110"
        if ((data[29] >= 0.150999993 and data[29] != 0) and (data[15] < 22.1100006 or data[15] == 0)):
            return "15"
        if ((data[29] >= 0.150999993 and data[29] != 0) and (data[15] >= 22.1100006 and data[15] != 0)):
            return "16"
    if i == 2:
        if ((data[9] < 0.0492300019 or data[9] == 0) and (data[23] < 29.7950001 or data[23] == 0) and (
                data[12] < 0.500550032 or data[12] == 0)):
            return "27"
        if ((data[9] < 0.0492300019 or data[9] == 0) and (data[23] < 29.7950001 or data[23] == 0) and (
                data[12] >= 0.500550032 and data[12] != 0)):
            return "28"
        if ((data[9] < 0.0492300019 or data[9] == 0) and (data[23] >= 29.7950001 and data[23] != 0) and (
                data[25] < 766.450012 or data[25] == 0) and (data[16] < 0.0072949999 or data[16] == 0)):
            return "213"
        if ((data[9] < 0.0492300019 or data[9] == 0) and (data[23] >= 29.7950001 and data[23] != 0) and (
                data[25] < 766.450012 or data[25] == 0) and (data[16] >= 0.0072949999 and data[16] != 0)):
            return "214"
        if ((data[9] < 0.0492300019 or data[9] == 0) and (data[23] >= 29.7950001 and data[23] != 0) and (
                data[25] >= 766.450012 and data[25] != 0)):
            return "210"
        if ((data[9] >= 0.0492300019 and data[9] != 0) and (data[23] < 23.7399998 or data[23] == 0) and (
                data[25] < 810.800049 or data[25] == 0)):
            return "211"
        if ((data[9] >= 0.0492300019 and data[9] != 0) and (data[23] < 23.7399998 or data[23] == 0) and (
                data[25] >= 810.800049 and data[25] != 0)):
            return "212"
        if ((data[9] >= 0.0492300019 and data[9] != 0) and (data[23] >= 23.7399998 and data[23] != 0)):
            return "26"
    if i == 3:
        if ((data[24] < 102.050003 or data[24] == 0) and (data[17] < 0.0120250005 or data[17] == 0) and (
                data[18] < 0.012205 or data[18] == 0)):
            return "37"
        if ((data[24] < 102.050003 or data[24] == 0) and (data[17] < 0.0120250005 or data[17] == 0) and (
                data[18] >= 0.012205 and data[18] != 0)):
            return "38"
        if ((data[24] < 102.050003 or data[24] == 0) and (data[17] >= 0.0120250005 and data[17] != 0) and (
                data[8] < 0.103699997 or data[8] == 0)):
            return "39"
        if ((data[24] < 102.050003 or data[24] == 0) and (data[17] >= 0.0120250005 and data[17] != 0) and (
                data[8] >= 0.103699997 and data[8] != 0)):
            return "310"
        if ((data[24] >= 102.050003 and data[24] != 0) and (data[28] < 0.207000002 or data[28] == 0)):
            return "35"
        if ((data[24] >= 102.050003 and data[24] != 0) and (data[28] >= 0.207000002 and data[28] != 0) and (
                data[3] < 16.6049995 or data[3] == 0)):
            return "311"
        if ((data[24] >= 102.050003 and data[24] != 0) and (data[28] >= 0.207000002 and data[28] != 0) and (
                data[3] >= 16.6049995 and data[3] != 0) and (data[30] < 0.254150003 or data[30] == 0)):
            return "313"
        if ((data[24] >= 102.050003 and data[24] != 0) and (data[28] >= 0.207000002 and data[28] != 0) and (
                data[3] >= 16.6049995 and data[3] != 0) and (data[30] >= 0.254150003 and data[30] != 0)):
            return "314"
    if i == 4:
        if ((data[25] < 929.800049 or data[25] == 0) and (data[26] < 0.140850008 or data[26] == 0) and (
                data[11] < 0.0565499999 or data[11] == 0)):
            return "47"
        if ((data[25] < 929.800049 or data[25] == 0) and (data[26] < 0.140850008 or data[26] == 0) and (
                data[11] >= 0.0565499999 and data[11] != 0) and (data[6] < 0.0993099958 or data[6] == 0)):
            return "411"
        if ((data[25] < 929.800049 or data[25] == 0) and (data[26] < 0.140850008 or data[26] == 0) and (
                data[11] >= 0.0565499999 and data[11] != 0) and (data[6] >= 0.0993099958 and data[6] != 0)):
            return "412"
        if ((data[25] < 929.800049 or data[25] == 0) and (data[26] >= 0.140850008 and data[26] != 0) and (
                data[3] < 19.4799995 or data[3] == 0)):
            return "49"
        if ((data[25] < 929.800049 or data[25] == 0) and (data[26] >= 0.140850008 and data[26] != 0) and (
                data[3] >= 19.4799995 and data[3] != 0)):
            return "410"
        if ((data[25] >= 929.800049 and data[25] != 0) and (data[7] < 0.0856299996 or data[7] == 0)):
            return "45"
        if ((data[25] >= 929.800049 and data[25] != 0) and (data[7] >= 0.0856299996 and data[7] != 0)):
            return "46"
    if i == 5:
        if ((data[29] < 0.0917700008 or data[29] == 0) and (data[27] < 0.0992100015 or data[27] == 0)):
            return "53"
        if ((data[29] < 0.0917700008 or data[29] == 0) and (data[27] >= 0.0992100015 and data[27] != 0)):
            return "54"
        if ((data[29] >= 0.0917700008 and data[29] != 0) and (data[23] < 25.6800003 or data[23] == 0) and (
                data[25] < 810.800049 or data[25] == 0)):
            return "57"
        if ((data[29] >= 0.0917700008 and data[29] != 0) and (data[23] < 25.6800003 or data[23] == 0) and (
                data[25] >= 810.800049 and data[25] != 0)):
            return "58"
        if ((data[29] >= 0.0917700008 and data[29] != 0) and (data[23] >= 25.6800003 and data[23] != 0)):
            return "56"
    if i == 6:
        if ((data[12] < 0.563899994 or data[12] == 0) and (data[30] < 0.282800019 or data[30] == 0) and (
                data[29] < 0.11135 or data[29] == 0)):
            return "65"
        if ((data[12] < 0.563899994 or data[12] == 0) and (data[30] < 0.282800019 or data[30] == 0) and (
                data[29] >= 0.11135 and data[29] != 0)):
            return "66"
        if ((data[12] < 0.563899994 or data[12] == 0) and (data[30] >= 0.282800019 and data[30] != 0) and (
                data[23] < 26.3699989 or data[23] == 0)):
            return "67"
        if ((data[12] < 0.563899994 or data[12] == 0) and (data[30] >= 0.282800019 and data[30] != 0) and (
                data[23] >= 26.3699989 and data[23] != 0)):
            return "68"
        if ((data[12] >= 0.563899994 and data[12] != 0)):
            return "62"
    if i == 7:
        if ((data[15] < 19.7900009 or data[15] == 0)):
            return "71"
        if ((data[15] >= 19.7900009 and data[15] != 0) and (data[28] < 0.19295001 or data[28] == 0)):
            return "73"
        if ((data[15] >= 19.7900009 and data[15] != 0) and (data[28] >= 0.19295001 and data[28] != 0) and (
                data[26] < 0.136050001 or data[26] == 0)):
            return "75"
        if ((data[15] >= 19.7900009 and data[15] != 0) and (data[28] >= 0.19295001 and data[28] != 0) and (
                data[26] >= 0.136050001 and data[26] != 0)):
            return "76"
    if i == 8:
        if ((data[5] < 694.5 or data[5] == 0) and (data[20] < 0.0174399987 or data[20] == 0)):
            return "83"
        if ((data[5] < 694.5 or data[5] == 0) and (data[20] >= 0.0174399987 and data[20] != 0)):
            return "84"
        if ((data[5] >= 694.5 and data[5] != 0)):
            return "82"
    if i == 9:
        if ((data[23] < 23.3499985 or data[23] == 0)):
            return "91"
        if ((data[23] >= 23.3499985 and data[23] != 0) and (data[17] < 0.0177149996 or data[17] == 0)):
            return "93"
        if ((data[23] >= 23.3499985 and data[23] != 0) and (data[17] >= 0.0177149996 and data[17] != 0)):
            return "94"
    if i == 10:
        if ((data[15] < 22.9400005 or data[15] == 0)):
            return "101"
        if ((data[15] >= 22.9400005 and data[15] != 0) and (data[28] < 0.252350003 or data[28] == 0)):
            return "103"
        if ((data[15] >= 22.9400005 and data[15] != 0) and (data[28] >= 0.252350003 and data[28] != 0)):
            return "104"
    if i == 11:
        if ((data[30] < 0.294449985 or data[30] == 0) and (data[10] < 0.165100008 or data[10] == 0)):
            return "113"
        if ((data[30] < 0.294449985 or data[30] == 0) and (data[10] >= 0.165100008 and data[10] != 0)):
            return "114"
        if ((data[30] >= 0.294449985 and data[30] != 0)):
            return "112"
    if i == 12:
        if ((data[6] < 0.0904249996 or data[6] == 0)):
            return "121"
        if ((data[6] >= 0.0904249996 and data[6] != 0)):
            return "122"
    if i == 13:
        if ((data[25] < 739.199951 or data[25] == 0)):
            return "131"
        if ((data[25] >= 739.199951 and data[25] != 0)):
            return "132"
    if i == 14:
        if ((data[3] < 20.1949997 or data[3] == 0)):
            return "141"
        if ((data[3] >= 20.1949997 and data[3] != 0)):
            return "142"
    if i == 15:
        if ((data[6] < 0.0994049981 or data[6] == 0)):
            return "151"
        if ((data[6] >= 0.0994049981 and data[6] != 0)):
            return "152"
    if i == 16:
        if ((data[15] < 31.2849998 or data[15] == 0)):
            return "161"
        if ((data[15] >= 31.2849998 and data[15] != 0)):
            return "162"
    if i == 17:
        if ((data[10] < 0.166350007 or data[10] == 0)):
            return "171"
        if ((data[10] >= 0.166350007 and data[10] != 0)):
            return "172"
    if i == 18:
        if ((data[28] < 0.216399997 or data[28] == 0)):
            return "181"
        if ((data[28] >= 0.216399997 and data[28] != 0)):
            return "182"
    if i == 19:
        if ((data[3] < 20.1549988 or data[3] == 0)):
            return "191"
        if ((data[3] >= 20.1549988 and data[3] != 0)):
            return "192"
    if i == 20:
        if ((data[20] < 0.0160499997 or data[20] == 0)):
            return "201"
        if ((data[20] >= 0.0160499997 and data[20] != 0)):
            return "202"
    if i == 21:
        if ((data[30] < 0.28065002 or data[30] == 0)):
            return "211"
        if ((data[30] >= 0.28065002 and data[30] != 0)):
            return "212"
    if i == 22:
        if ((data[25] < 739.199951 or data[25] == 0)):
            return "221"
        if ((data[25] >= 739.199951 and data[25] != 0)):
            return "222"
    if i == 23:
        if ((data[17] < 0.01767 or data[17] == 0)):
            return "231"
        if ((data[17] >= 0.01767 and data[17] != 0)):
            return "232"
    if i == 24:
        if ((data[21] < 0.00267850002 or data[21] == 0)):
            return "241"
        if ((data[21] >= 0.00267850002 and data[21] != 0)):
            return "242"
    if i == 25:
        if ((data[29] < 0.111000001 or data[29] == 0)):
            return "251"
        if ((data[29] >= 0.111000001 and data[29] != 0)):
            return "252"
    if i == 26:
        if ((data[17] < 0.01767 or data[17] == 0)):
            return "261"
        if ((data[17] >= 0.01767 and data[17] != 0)):
            return "262"
    if i == 27:
        if ((data[15] < 26.4350014 or data[15] == 0)):
            return "271"
        if ((data[15] >= 26.4350014 and data[15] != 0)):
            return "272"
    if i == 28:
        if ((data[20] < 0.0157900006 or data[20] == 0)):
            return "281"
        if ((data[20] >= 0.0157900006 and data[20] != 0)):
            return "282"
    if i == 29:
        if ((data[29] < 0.111000001 or data[29] == 0)):
            return "291"
        if ((data[29] >= 0.111000001 and data[29] != 0)):
            return "292"
    if i == 30:
        if ((data[3] < 19.3149986 or data[3] == 0)):
            return "301"
        if ((data[3] >= 19.3149986 and data[3] != 0)):
            return "302"
    if i == 31:
        if ((data[21] < 0.00274749985 or data[21] == 0)):
            return "311"
        if ((data[21] >= 0.00274749985 and data[21] != 0)):
            return "312"
    if i == 32:
        if ((data[17] < 0.01767 or data[17] == 0)):
            return "321"
        if ((data[17] >= 0.01767 and data[17] != 0)):
            return "322"
    if i == 33:
        if ((data[29] < 0.111000001 or data[29] == 0)):
            return "331"
        if ((data[29] >= 0.111000001 and data[29] != 0)):
            return "332"
    if i == 34:
        if ((data[10] < 0.172349989 or data[10] == 0)):
            return "341"
        if ((data[10] >= 0.172349989 and data[10] != 0)):
            return "342"
    if i == 35:
        if ((data[30] < 0.269600004 or data[30] == 0)):
            return "351"
        if ((data[30] >= 0.269600004 and data[30] != 0)):
            return "352"
    if i == 36:
        if ((data[10] < 0.172349989 or data[10] == 0)):
            return "361"
        if ((data[10] >= 0.172349989 and data[10] != 0)):
            return "362"
    if i == 37:
        if ((data[21] < 0.00275600003 or data[21] == 0)):
            return "371"
        if ((data[21] >= 0.00275600003 and data[21] != 0)):
            return "372"
    if i == 38:
        if ((data[22] < 15.6399994 or data[22] == 0)):
            return "381"
        if ((data[22] >= 15.6399994 and data[22] != 0)):
            return "382"
    if i == 39:
        if ((data[17] < 0.017310001 or data[17] == 0)):
            return "391"
        if ((data[17] >= 0.017310001 and data[17] != 0)):
            return "392"
    if i == 40:
        if ((data[22] < 15.6399994 or data[22] == 0)):
            return "401"
        if ((data[22] >= 15.6399994 and data[22] != 0)):
            return "402"
    return "0"


def XGBprocess35(data,i):
    if i == 0:
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] < 0.135049999 or data[29] == 0) and (
                data[15] < 47.0349998 or data[15] == 0)):
            return 7
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] < 0.135049999 or data[29] == 0) and (
                data[15] >= 47.0349998 and data[15] != 0)):
            return 8
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] >= 0.135049999 and data[29] != 0) and (
                data[3] < 20.2999992 or data[3] == 0) and (data[8] < 0.119749993 or data[8] == 0)):
            return 15
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] >= 0.135049999 and data[29] != 0) and (
                data[3] < 20.2999992 or data[3] == 0) and (data[8] >= 0.119749993 and data[8] != 0)):
            return 16
        if ((data[24] < 105.949997 or data[24] == 0) and (data[29] >= 0.135049999 and data[29] != 0) and (
                data[3] >= 20.2999992 and data[3] != 0)):
            return 10
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] < 20.6450005 or data[23] == 0) and (
                data[22] < 17.7399998 or data[22] == 0)):
            return 11
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] < 20.6450005 or data[23] == 0) and (
                data[22] >= 17.7399998 and data[22] != 0)):
            return 12
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] < 0.0488649979 or data[9] == 0) and (data[17] < 0.0171750002 or data[17] == 0)):
            return 17
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] < 0.0488649979 or data[9] == 0) and (data[17] >= 0.0171750002 and data[17] != 0)):
            return 18
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] >= 0.0488649979 and data[9] != 0) and (data[6] < 0.0827400014 or data[6] == 0)):
            return 19
        if ((data[24] >= 105.949997 and data[24] != 0) and (data[23] >= 20.6450005 and data[23] != 0) and (
                data[9] >= 0.0488649979 and data[9] != 0) and (data[6] >= 0.0827400014 and data[6] != 0)):
            return 20
        return 0
    return 0





# if i == 0:
#     if((data[24]<105.949997 or data[24] ==0) and (data[29]<0.135049999 or data[29] ==0) and (data[15]<47.0349998 or data[15] ==0)):
#              return 7
#     if((data[24]<105.949997 or data[24] ==0) and (data[29]<0.135049999 or data[29] ==0) and (data[15]>=47.0349998 and data[15] !=0)):
#              return 8
#     if((data[24]<105.949997 or data[24] ==0) and (data[29]>=0.135049999 and data[29] !=0) and (data[3]<20.2999992 or data[3] ==0) and (data[8]<0.119749993 or data[8] ==0)):
#              return 15
#     if((data[24]<105.949997 or data[24] ==0) and (data[29]>=0.135049999 and data[29] !=0) and (data[3]<20.2999992 or data[3] ==0) and (data[8]>=0.119749993 and data[8] !=0)):
#              return 16
#     if((data[24]<105.949997 or data[24] ==0) and (data[29]>=0.135049999 and data[29] !=0) and (data[3]>=20.2999992 and data[3] !=0)):
#              return 10
#     if((data[24]>=105.949997 and data[24] !=0) and (data[23]<20.6450005 or data[23] ==0) and (data[22]<17.7399998 or data[22] ==0)):
#              return 11
#     if((data[24]>=105.949997 and data[24] !=0) and (data[23]<20.6450005 or data[23] ==0) and (data[22]>=17.7399998 and data[22] !=0)):
#              return 12
#     if((data[24]>=105.949997 and data[24] !=0) and (data[23]>=20.6450005 and data[23] !=0) and (data[9]<0.0488649979 or data[9] ==0) and (data[17]<0.0171750002 or data[17] ==0)):
#              return 17
#     if((data[24]>=105.949997 and data[24] !=0) and (data[23]>=20.6450005 and data[23] !=0) and (data[9]<0.0488649979 or data[9] ==0) and (data[17]>=0.0171750002 and data[17] !=0)):
#              return 18
#     if((data[24]>=105.949997 and data[24] !=0) and (data[23]>=20.6450005 and data[23] !=0) and (data[9]>=0.0488649979 and data[9] !=0) and (data[6]<0.0827400014 or data[6] ==0)):
#              return 19
#     if((data[24]>=105.949997 and data[24] !=0) and (data[23]>=20.6450005 and data[23] !=0) and (data[9]>=0.0488649979 and data[9] !=0) and (data[6]>=0.0827400014 and data[6] !=0)):
#              return 20
#     return 0
# if i == 1:
#     if((data[29]<0.150999993 or data[29] ==0) and (data[25]<957.450012 or data[25] ==0) and (data[3]<21.4349995 or data[3] ==0) and (data[13]<2.09749985 or data[13] ==0)):
#              return 11
#     if((data[29]<0.150999993 or data[29] ==0) and (data[25]<957.450012 or data[25] ==0) and (data[3]<21.4349995 or data[3] ==0) and (data[13]>=2.09749985 and data[13] !=0)):
#              return 12
#     if((data[29]<0.150999993 or data[29] ==0) and (data[25]<957.450012 or data[25] ==0) and (data[3]>=21.4349995 and data[3] !=0) and (data[22]<14.4300003 or data[22] ==0)):
#              return 13
#     if((data[29]<0.150999993 or data[29] ==0) and (data[25]<957.450012 or data[25] ==0) and (data[3]>=21.4349995 and data[3] !=0) and (data[22]>=14.4300003 and data[22] !=0) and (data[30]<0.269600004 or data[30] ==0)):
#              return 15
#     if((data[29]<0.150999993 or data[29] ==0) and (data[25]<957.450012 or data[25] ==0) and (data[3]>=21.4349995 and data[3] !=0) and (data[22]>=14.4300003 and data[22] !=0) and (data[30]>=0.269600004 and data[30] !=0)):
#              return 16
#     if((data[29]<0.150999993 or data[29] ==0) and (data[25]>=957.450012 and data[25] !=0) and (data[28]<0.222950011 or data[28] ==0)):
#              return 9
#     if((data[29]<0.150999993 or data[29] ==0) and (data[25]>=957.450012 and data[25] !=0) and (data[28]>=0.222950011 and data[28] !=0)):
#              return 10
#     if((data[29]>=0.150999993 and data[29] !=0) and (data[15]<22.1100006 or data[15] ==0)):
#              return 5
#     if((data[29]>=0.150999993 and data[29] !=0) and (data[15]>=22.1100006 and data[15] !=0)):
#              return 6
#     return 0

def XGBprocess4(data):
    s=[""]*40
    for i in range(40):
        s[i] = XGBprocess35(data,i)
    return s
