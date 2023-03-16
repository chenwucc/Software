#!/usr/bin/env python
# coding: utf-8

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import glob

def convert_to_datetime(intime):

    intimearray = np.array(intime, dtype='float64')
    outtime = []
    base = dt.datetime(year = 1980, month = 1, day = 6)
    for t in intimearray:
        outtime.append(base + dt.timedelta(seconds = t))

    return outtime



def sep_prof_1tel(tp_alt,rec_index,sc_sza):
    ### seperate altitude profiles for 1 tel --------------------------

    llen = len(tp_alt)
    index2 = np.arange(llen-1)+1
    index1 = np.arange(llen-1)

    index_all = np.arange(llen)

    lp = sc_sza>=90.0
    alt1 = tp_alt[lp]
    rec1 = rec_index[lp]
    sza1 = sc_sza[lp]
    index_n = index_all[lp]

    index2_1 = index_n[1:len(index_n)]
    index1_1 = index_n[0:-1]

    dalt1 = alt1[1:len(index_n)]-alt1[0:-1]
    fp = -dalt1 > 4
    ffp = index1_1<=index1_1[fp][0]
    index_sn = np.concatenate((index1_1[ffp][0],index2_1[fp]),axis=None)

    ffp = index2_1>=index2_1[fp][-1]
    index_en = np.concatenate((index1_1[fp],index2_1[ffp][-1]),axis=None)

    lp = sc_sza<90.0
    alt1 = tp_alt[lp]
    rec1 = rec_index[lp]
    sza1 = sc_sza[lp]
    index_d = index_all[lp]
    index2_1 = index_d[1:len(index_d)]
    index1_1 = index_d[0:-1]

    dalt1 = alt1[1:len(index_d)]-alt1[0:-1]
    fp = -dalt1 > 10.0
    ffp = index1_1<=index1_1[fp][0]
    index_sd = np.concatenate((index1_1[ffp][0],index2_1[fp]),axis=None)
    ffp = index2_1>=index2_1[fp][-1]
    index_ed = np.concatenate((index1_1[fp],index2_1[ffp][-1]),axis=None)

    flag = np.ones(len(rec_index))*np.nan
    prf = np.ones(len(rec_index))*np.nan

    flag[index_ed] = 1
    flag[index_en] = 1

    index_ee = np.concatenate((index_ed,index_en),axis=None)
    index_ee = np.sort(index_ee, axis=None)

    ddalt = tp_alt[index_ee[0:-1]+1]-tp_alt[index_ee[0:-1]]
    llp = abs(ddalt)<4

    if len(ddalt[llp])>0:
        flag[index_ee[0:-1][llp]] = 2

    #print('flag',np.nanmin(flag),np.nanmax(flag))

    prf[0] = 0
    for k11,k1 in enumerate(flag[0:-1]):

        if k1 != 1:

            prf[k11+1] = prf[k11]
        else:
            prf[k11+1] = prf[k11-1]+1

    return prf



def read_los_prf(ifile):

    ### -----------------------------------------------
    #- seperate by telescopes
    #- label orbit, altitude profile, and warm/cold side
    #- contains:
    #  1. sep_prof_1tel
    #  2. convert_to_datetime

    ncfile = Dataset(ifile, 'r')

    fw_config = np.array(ncfile.variables['fw_config'])
    alltimes = np.array(ncfile.variables['time'][:], dtype='float64')

    angles = [45, 135, 225, 315, 405]
    tel_id = np.array(ncfile.variables['tel_id'][:])
    tp_lat = np.array(ncfile.variables['tp_lat'][:])
    tp_lon = np.array(ncfile.variables['tp_lon'][:])
    tp_alt = np.array(ncfile.variables['tp_alt'][:])

    sc_lon = np.array(ncfile.variables['sc_lon'][:])

    s_los = np.array(ncfile.variables['s'][:])
    s_var = np.array(ncfile.variables['var_s'][:])
    p_status = np.array(ncfile.variables['p_status'][:])
    table_index = np.array(ncfile.variables['table_index'][:])
    los_dir = np.array(ncfile.variables['los_direction'][:])


    b_los = np.array(ncfile.variables['b'][:])
    b_var = np.array(ncfile.variables['var_b'][:])

    asc_flag = np.array(ncfile.variables['ascending'][:])
    asc_flag = asc_flag.astype('U13')
    data_ok = np.array(ncfile.variables['data_ok'][:])
    data_ok = data_ok.astype('U13')
    in_saa = np.array(ncfile.variables['in_saa'][:])
    in_saa = in_saa.astype('U13')

    sc_sza = np.array(ncfile.variables['sc_sza'][:])
    tp_sza = np.array(ncfile.variables['tp_sza'][:])
    tp_sscat = np.array(ncfile.variables['tp_sscat'][:])

    rec_index = np.array(ncfile.variables['rec_index'][:])
    b_angle = ncfile.getncattr('solar_beta_angle')

    flight_dir = np.array(ncfile.variables['flight_dir'][:])
    flight_dir = flight_dir.astype('U13')
    tp_lst = np.array(ncfile.variables['tp_lst'][:])
    tp_track = np.array(ncfile.variables['tp_track'][:])

    #print('flight_dir F',flight_dir[flight_dir=='F'])
    #print('flight_dir B',flight_dir[flight_dir=='B'])

    times = {}
    tplon = {}
    tplat = {}
    tpalt = {}
    los = {}
    blos = {}
    bvar = {}
    dir = {}
    var = {}
    ascflag = {}
    flightdir = {}

    insaa = {}

    scsza = {}
    tpsza = {}
    tpsscat = {}
    recindex = {}
    orbid = {}
    prfid = {}
    warm = {}

    tplst = {}
    tptrack = {}

    ### orb seperate
    orb0 = np.ones(len(rec_index))*np.nan
    lp = tp_alt>0
    orb = orb0[lp]
    asc_tmp = asc_flag[lp]
    orb[0] = 0
    for k11,k1 in enumerate(asc_tmp):
        if k11==0:
            continue

        if k1 == asc_tmp[k11-1]:
            orb[k11] = orb[k11-1]
        else:
            orb[k11] = orb[k11-1]+1
    orb0[lp] = orb
    asc_flag = asc_flag.flatten()
    fp = asc_flag=='T'
    tmp = orb0[fp]
    orb0[fp] = (tmp-np.nanmin(tmp))/2

    fp = asc_flag=='F'
    tmp = orb0[fp]
    orb0[fp] = (tmp-np.nanmin(tmp))/2

    ### alt profile seperate
    prf0 = np.ones(rec_index[-1])*np.nan
    for k11,a in enumerate(angles[0:4]):
        lp = (tel_id==a)&(tp_alt>0)
        prfs = sep_prof_1tel(tp_alt[lp],rec_index[lp],
                               sc_sza[lp])
        prf0[lp] = prfs

    ## warm- and cold- side
    warm_side = np.ones(len(tel_id))*np.nan
    warm_side[tel_id==45]    = np.zeros(len(tel_id[tel_id==45]))
    warm_side[tel_id==135]   = np.zeros(len(tel_id[tel_id==135]))
    warm_side[tel_id==225]   = np.ones(len(tel_id[tel_id==225]))
    warm_side[tel_id==315]   = np.ones(len(tel_id[tel_id==315]))

    data_ok = data_ok[:,0]
    for a in angles[0:4]:
        tmp      = alltimes[(sc_lon>=0) & (tel_id == a) &    (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        times[a] = convert_to_datetime(tmp)
        tplon[a] = tp_lon[(sc_lon>=0) & (tel_id == a) &      (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        tplat[a] = tp_lat[(sc_lon>=0) & (tel_id == a) &      (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        tpalt[a] = tp_alt[(sc_lon>=0) & (tel_id == a) &      (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        los[a]    = s_los[(sc_lon>=0) & (tel_id == a) &      (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        blos[a]    = b_los[(sc_lon>=0) & (tel_id == a) &     (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        bvar[a]    = b_var[(sc_lon>=0) & (tel_id == a) &     (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        dir[a]  = los_dir[(sc_lon>=0) & (tel_id == a) &      (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        var[a] = np.sqrt(s_var[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')])
        ascflag[a]  = asc_flag[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        insaa[a]  =     in_saa[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]

        flightdir[a] = flight_dir[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]

        scsza[a]      = sc_sza[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        tpsza[a]      = tp_sza[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        tpsscat[a]  = tp_sscat[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        orbid[a]     = orb0[(sc_lon>=0) & (tel_id == a) &    (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        prfid[a]     = prf0[(sc_lon>=0) & (tel_id == a) &    (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        warm[a] = warm_side[(sc_lon>=0) & (tel_id == a) &    (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        tplst[a]  = tp_lst[(sc_lon>=0) & (tel_id == a) &     (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]
        tptrack[a]  = tp_track[(sc_lon>=0) & (tel_id == a) & (p_status == 0) & (fw_config == 5) & (data_ok == 'T')]

    ncfile.close()
    data = {'time' : times,
            'tplon' : tplon,
            'tplat' : tplat,
            'tpalt' : tpalt,
            'los' : los,
            'blos' : blos,
            'bvar' : bvar,
            'dir' : dir,
            'ascflag': ascflag,
            'insaa': insaa,

            'flightdir': flightdir,

            'scsza': scsza,
            'tpsza': tpsza,
            'tpsscat': tpsscat,
            'tplst': tplst,
            'tptrack': tptrack,

            'var' : var,
            'orbid':orbid,
            'prfid':prfid,
            'warm':warm}
    return data



def get_1tel_df(tididata,angle):

    ### creat dataframe -------------------

    iS = 0
    iE = len(tididata["time"][angle])-1

    tplon   = tididata["tplon"][angle][iS:iE]
    tplat   = tididata["tplat"][angle][iS:iE]
    tpalt   = tididata["tpalt"][angle][iS:iE]
    los     = tididata["los"][angle][iS:iE]
    blos     = tididata["blos"][angle][iS:iE]
    bvar     = tididata["bvar"][angle][iS:iE]
    dir     = tididata["dir"][angle][iS:iE]
    var     = tididata["var"][angle][iS:iE]
    times   = tididata["time"][angle][iS:iE]
    ascflag = tididata["ascflag"][angle][iS:iE]
    tpsza   = tididata["tpsza"][angle][iS:iE]
    scsza   = tididata["scsza"][angle][iS:iE]
    tpsscat = tididata["tpsscat"][angle][iS:iE]

    flightdir = tididata["flightdir"][angle][iS:iE]

    orbid   = tididata["orbid"][angle][iS:iE]
    prfid   = tididata["prfid"][angle][iS:iE]
    warm    = tididata["warm"][angle][iS:iE]

    tplst = tididata["tplst"][angle][iS:iE]
    tptrack = tididata["tptrack"][angle][iS:iE]
    insaa = tididata["insaa"][angle][iS:iE]

    times = pd.to_datetime(times)

    df = pd.DataFrame(list(zip(
        times ,
        tplon ,
        tplat ,
        tpalt ,
        los  ,
        blos  ,
        bvar  ,
        dir  ,
        ascflag,
        flightdir,
        tpsza,
        scsza,
        tpsscat,
        tplst,
        tptrack,
        insaa,

        var,
        orbid,
        prfid,
        warm
        )),columns=[
        'time' ,
        'tplon' ,
        'tplat' ,
        'tpalt' ,
        'los',
        'blos',
        'bvar',
        'dir',
        'ascflag',
        'flightdir',
        'tpsza',
        'scsza',
        'tpsscat',
        'tplst',
        'tptrack',
        'insaa',

        'var',
        'orbid',
        'prfid',
        'warm'])

    return df



def plot_1prf(df,telid):

    iTs = df['time']
    #print(iTs)
    iT = iTs.iloc[0]
    year = iT.year
    doy = iT.timetuple().tm_yday

    alt = np.asarray(df['tpalt'].astype(int))
    #print(alt)
    if len(alt) == len(np.unique(alt)):
        f_signal = 'good'
        signal = 1
    else:
        f_signal = 'bad'
        signal = 0

    #--- plot tidi --------------------------
    fig = plt.figure(figsize=(8,10))
    ax11 = fig.add_subplot(5,2,1)
    ax11.plot(df['los'],df['tpalt'],'-o',label='TIDI',
            alpha=0.75)
    ax11.text(0.05,1.01,
            iT.strftime('%Y-%m-%d %H:%M:%S'),
            transform=ax11.transAxes,
            fontsize=8)
    ax11.set_ylabel('Alt (km)')
    ax11.set_xlabel('los (m/s)')
    ax11.text(0.85,1.01,
            'tel: {:3d}'.format(telid),
            transform=ax11.transAxes,
            fontsize=8)


    ax12 = fig.add_subplot(5,2,2)
    ax12.errorbar(df['blos'],df['tpalt'],
                xerr = np.sqrt(df['bvar']),
                marker='o',
                #markersize=8,
                #elinewidth=5,
                alpha=0.75,
                linestyle='')
    ax12.set_ylabel('Alt (km)')
    ax12.set_xlabel('blos (R)')
    ax12.text(0.85,1.01,
            f_signal,
            transform=ax12.transAxes,
            fontsize=8)

    ax2 = fig.add_subplot(5,1,2)
    ax2.plot(df['time'],df['tplat'],'-o',
            alpha=0.75)
    ax2.set_ylabel('tplat (Deg)')

    ax3 = fig.add_subplot(5,1,3)
    ax3.plot(df['time'],df['tplon'],'-o',
            alpha=0.75)
    ax3.set_ylabel('tplon (Deg)')

    ax4 = fig.add_subplot(5,1,4)
    ax4.plot(df['time'],df['tplst'],'-o',
            alpha=0.75)

    ax4.set_ylabel('tplst (hr)')
    ax4.set_xlabel('time')

    ax5 = fig.add_subplot(5,1,5)
    ax5.plot(df['time'],df['tpalt'],'-o',
            alpha=0.75)

    ax5.set_ylabel('tpalt (km)')
    ax5.set_xlabel('time')
    ax5.grid(True)

    plt.tight_layout()
    fig.savefig(savepath + 'tidi_red_prf_'
                + iT.strftime('%Y%m%d_%H%M')
                + '_tel{:03d}.png'.format(telid))
    plt.close(fig)
    #plt.show()

    return signal



def process_prfs(df,telid):
    ### --------------------------------------
    #   process each altitude profile

    lp = (abs(df['los'])<500)
    df = df[lp]

    prfid = np.asarray(df['prfid'])
    tmp = set(prfid)

    iTs = df['time']
    iT = iTs.iloc[500]
    year = iT.year
    doy = iT.timetuple().tm_yday

    signal = np.ones(len(tmp)) * np.nan
    for k11,k1 in enumerate(tmp):

        lp = prfid==k1
        dft = df[lp]
        time = dft['time'].astype('datetime64[ns]').quantile(0.5,
                interpolation="midpoint")

        if len(dft[dft['tpalt']>200])<5:  # if measurements < 5, ignore this prf
            continue

        signal[k11] = plot_1prf(dft.copy(),telid)
    nums = [len(signal),
            len(signal[signal==signal]),
            len(signal[signal==0]),
            len(signal[signal==1])]
    return nums



def main(year,doys):

    tels = [45,135,225,315]

    for i in doys:

        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()

        dirr = datadir +'/TIDI_PB_{:04d}{:03d}_*'.format(
                int(year),int(i)) + ver + '*.LOS'
        file1 = glob.glob(dirr)
        if len(file1)==0:
            continue

        date = file1[0][-32:-25]
        print('reading...',date,file1[0])

        tididata = read_los_prf(file1[0])

        for telid in tels:

            df01 = get_1tel_df(tididata,telid)
            nums = process_prfs(df01,telid)

            print('-->>> summary for tel{:3d}'.format(telid))
            print('Total prfs: ', nums[1],'/',nums[0])
            print('Bad prfs: ', nums[2])
            print('Good prfs: ', nums[3])

    return

if __name__ == '__main__':


    ver = 'D011'                 # data version

    year = 2021                  # set year
    doys = np.arange(308,309,1)  # set day of year
    savepath = './plots/' # set savepath for output plots
    datadir  = './data/'  # set datadir for LOS files

    main(year,doys)


