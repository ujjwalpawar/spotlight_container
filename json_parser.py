import os
import pandas as pd
import json
import pprint
import numpy as np
import warnings
import matplotlib.pyplot as plt
import datetime
import os
import math


def get_variance(value, bin, mean):
    var = []
    for i in range(len(value)):
        var.append(((value[i] - mean)**2)*bin[i])
    return sum(var)/sum(bin) if sum(bin) != 0 else 0

# Function to calculate skewness
def get_skewness(data):
    return float(np.mean((data - np.mean(data))**3) / (np.std(data)**3))

# Function to calculate kurtosis
def get_kurtosis(data):
    return float(np.mean((data - np.mean(data))**4) / (np.std(data)**4))
def get_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return iqr
def detect_outliers_mean_std(data, mean, std, num_std=2):
    lower_bound = mean - num_std * std
    upper_bound = mean + num_std * std
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return np.mean(outliers)


Hook_Vs_Codelet= {'mac_sinr_update': 'sinr_stats',
 'fapi_gnb_ul_config_req': 'fapi_gnb_ul_config_stats',
 'midhaul_rx': 'mhrx_ps',
 'mac_dl_bo_update': 'bo_stats',
 'fapi_gnb_dl_config_req': 'fapi_gnb_dl_config_stats',
 'backhaul_rx': 'bhrx_ps',
 'f1u_rlc_size': 'dfr_ps',
 'backhaul_tx': 'bhtx_ps',
 'rlc_mac_size': 'drm_ps',
 'mac_ul_crc_ind': 'crc_stats',
 'mac_rlc_size': 'umr_ps',
 'mac_bsr_update': 'bsr_stats',
 'rlc_f1u_size': 'urf_ps',
 'mac_csi_report': 'csi_stats',
 'mac_dl_harq': 'harq_dl_stats',
 'midhaul_tx': 'mhtx_ps'}

class parser:
    hist_shift ={'bhrx_ps' : 8 ,
                'bhtx_ps' : 8,
                'mhrx_ps' : 8,
                'mhtx_ps' : 8,
                'f1u_rlc_size' : 8,
                'fapi_gnb_dl_config_req_prb' : 5,
                'fapi_gnb_dl_config_req_mcs' : 2,
                'fapi_gnb_dl_config_req_tbs' : 13,
                'fapi_gnb_ul_config_req_prb' : 5,
                'fapi_gnb_ul_config_req_mcs' : 2,
                'fapi_gnb_ul_config_req_tbs' : 13,
                'mac_bsr_update' : 14,
                'mac_csi_report' : 2,
                'mac_dl_bo_update' : 8,
                'mac_dl_harq' : 0,
                'mac_rlc_size' : 8,
                'mac_sinr_update' : 5,
                'mac_ul_crc_ind' : 0,
                'rlc_f1u_size' : 8,
                'rlc_mac_size' : 8}
    def __init__(self):
        # self.filename = filename
        # self.timestamp = timestamp
        self.fig, self.axes = plt.subplots(nrows=3, ncols=3)
    # helper fucntions
    # get count given the key for histogram
    def cnt_calculator(self,inpt, key, valu, cnt_key):
        cnt=0
        for j in inpt:
            cnt+=j[cnt_key]
        return cnt
    # get avergae given the for histogram
    def calc_avg(self, bin,shift, min, max):
        total = 0
        BUCKET = 2**shift
        hist_value = [0] * 9
        cnt = 0
        for i, bin_count in enumerate(bin):
            if bin_count > 0:
                cnt+=1
                if(BUCKET*i<=min):
                    range_start = min
                else:
                    range_start = BUCKET*i
                if(BUCKET*(i + 1) - 1> max):
                    range_end = max
                else:
                    range_end = BUCKET*(i + 1) - 1
                range_size = range_end - range_start + 1
                items_per_bin = bin_count / range_size
                # Calculate total for this bin using formulas
                total += ((range_start + range_end) * range_size / 2)*items_per_bin
                # hist_value[i] = total
                # total=0
        return total/sum(bin)
    # get creates the histogram in array format
    def avg_packet_calc_hist(self, inpt, key, shift, min, max):
        hist = [0]*9
        for i in inpt:
            hist[i[key]]+=i['cnt']
        return self.calc_avg(hist,shift,min,max)
    # gets minimum value from the histogram
    def get_min(self, data):
        min_val = 99999999
        for i in data:
            if(i['val']<min_val):
                min_val = i['val']
        return min_val

    # gets maximum value from the histogram
    def get_max(self,data):
        max_val = 0
        for i in data:
            if(i['val']>max_val):
                max_val = i['val']
        return max_val
    # plot the histogram
    def show_plot(self):
        self.fig
    # calculate the runtime for platfrom kpi histogram
    def calc_runtime(self, bin):
        total = 0
        for i, bin_count in enumerate(bin):
            if bin_count > 0:
                range_start = 2 ** i
                range_end = 2 ** (i + 1) - 1
                range_size = range_end - range_start + 1
                items_per_bin = bin_count / range_size
                # Calculate total for this bin using formulas
                total += items_per_bin * ((range_start + range_end) * range_size/2)

        return total

    # calculate the hist stats for platfrom kpi histogram
    def calc_platform_hist(self, bin):
        total = 0
        value = []
        for i, bin_count in enumerate(bin):
            # print ("i, bin_count, bin:",i, bin_count, bin)
            if bin_count > 0:
                range_start = 2 ** i
                range_end = 2 ** (i + 1) - 1
                range_size = range_end - range_start + 1
                items_per_bin = bin_count / range_size
                # Calculate total for this bin using formulas
                value = np.append(value,((range_start + range_end) * range_size/2)*items_per_bin)
                total += items_per_bin * ((range_start + range_end) * range_size/2)


        mean = sum(value)/ sum(bin) if sum(bin) != 0 else 0
        var  = get_variance(value, bin, mean)
        std = var**0.5
        skewness = get_skewness(bin)
        kurtosis = get_kurtosis(bin)
        irq = get_iqr(bin)
        outliers = detect_outliers_mean_std(value, mean, std)
        return mean, var ,std, skewness, kurtosis, irq,  outliers

    # helper fuction calculate the hist stats for ran kpi histogram
    def get_procesesd_histogram(self, bin,shift, min, max):
        total = 0
        BUCKET = 2**shift
        hist_value = [0] * 9
        if len(bin) > 9:
          hist_value = [0] * len(bin)
        cnt = 0
        value = []
        # print ('bin:',bin)
        for i, bin_count in enumerate(bin):

            if bin_count > 0:
                cnt+=1
                if(BUCKET*i<=min):
                    range_start = min
                else:
                    range_start = BUCKET*i
                if(BUCKET*(i + 1) - 1> max):
                    range_end = max
                else:
                    range_end = BUCKET*(i + 1) - 1
                range_size = range_end - range_start + 1
                # print ('range_size, range_end,  range_start:',range_size, range_end,  range_start)
                items_per_bin = bin_count / range_size
                # Calculate total for this bin using formulas
                value = np.append(value,(((range_start + range_end) * range_size / 2)*items_per_bin))
                total += ((range_start + range_end) * range_size / 2)*items_per_bin
                # hist_value[i] = total
                # total=0
            else:
                value = np.append(value,0)
        # print ('value:', value)
        mean = sum(value)/ sum(bin)
        var  = get_variance(value, bin, mean)
        std = var**0.5
        skewness = get_skewness(bin)
        kurtosis = get_kurtosis(bin)
        irq = get_iqr(bin)
        outliers = detect_outliers_mean_std(value, mean, std)
        return mean, var ,std, skewness, kurtosis, irq,  outliers


    # fuction calculate the hist stats for ran kpi histogram
    def process_packet_hist(self, inpt, key, shift, min, max_in):
        hist = [0]*9

        # BO DL case
        max_key_value = max(inpt, key=lambda x: x[key])[key]
        if max_key_value + 1 > 9:
          hist = [0] * (max_key_value + 1)

        # print ('max_key_value:',max_key_value)
        for i in inpt:
            # print ("inpt, key, len(hist):",inpt, key, len(hist))
            # print(i[key])
            hist[i[key]]+=i['cnt']

        return self.get_procesesd_histogram(hist,shift,min,max_in)



    # fuctions to parse the hooks

    # parse f1u rlc size { kpi extracted : timestamp, cellId, stream_sn, stream_sn2, size, min, max, mean, range, std_dev, Interquartile range, Variance, skewness, kurtosis, outliers}
    def parse_f1u_rlc_size(self, json_object):
        # self.axes[0,0].set_title('f1u_rlc_size : dfr_agg')
        df=pd.DataFrame()
        data =[]
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']=='f1u_rlc_size'):
                tmp= json.loads(t['stream_payload_msg'])
                holder={}
                # pprint.pprint(tmp)
                batch=tmp['batch']
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                k=0
                holder['f1u_rlc_size']=0
                for k in range(len(batch)):
                    holder['f1u_rlc_size']+=batch[k]['size']
                holder['stream_sn']=t['stream_sn']
                holder['stream_sn2']=t['stream_sn2']
                holder['anomaly']=0
        #=============hisotgram calc==============================
                max = self.get_max(tmp['packetSizeMax'])
                min = self.get_min(tmp['packetSizeMin'])
                min_max_range = max - min
                hist__values = tmp['packetSizeHist']
                hist__key = 'packetSize'
                hist__shift = self.hist_shift['f1u_rlc_size']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['f1u_rlc_size_max'] = max
                holder['f1u_rlc_size_min'] = min
                holder['f1u_rlc_size_range'] = min_max_range
                holder['f1u_rlc_size_mean'] = mean
                holder['f1u_rlc_size_var'] = var
                holder['f1u_rlc_size_std'] = std
                holder['f1u_rlc_size_skewness'] = skewness
                holder['f1u_rlc_size_kurtosis'] = kurtosis
                holder['f1u_rlc_size_irq'] = irq
                holder['f1u_rlc_size_outliers'] = outliers
                    # =================================================
                data.append(holder)
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1

    # parse mac dl bo update
    def parse_mac_dl_bo_update(self, json_object):
        # self.axes[0,0].set_title('mac_dl_bo_update')
        df=pd.DataFrame()
        data =[]

        t=json_object
        try:
            if(t['stream_context_info']['hook_name']=='mac_dl_bo_update'):
                tmp= json.loads(t['stream_payload_msg'])
                # pprint.pprint(tmp)
                # break;
                holder={}
                # batch=tmp['batch']
                k=0
                holder['cellId']=tmp['cellId']
                holder['timestamp']=tmp['timestamp']
                # holder['Mac_dl_bo_size']=0
                # for k in range(len(batch)):
                #     holder['Mac_dl_bo_size']+=batch[k]['size']
                holder['stream_sn']=t['stream_sn']
            #=============hisotgram calc==============================
                max = self.get_max(tmp['l2BoMax'])
                min = self.get_min(tmp['l2BoMin'])
                min_max_range = max - min
                hist__values = tmp['l2BoHist']
                hist__key = 'queueLoad'
                hist__shift = self.hist_shift['mac_dl_bo_update']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['mac_dl_bo_max'] = max
                holder['mac_dl_bo_min'] = min
                holder['mac_dl_bo_range'] = min_max_range
                holder['mac_dl_bo_mean'] = mean
                holder['mac_dl_bo_var'] = var
                holder['mac_dl_bo_std'] = std
                holder['mac_dl_bo_skewness'] = skewness
                holder['mac_dl_bo_kurtosis'] = kurtosis
                holder['mac_dl_bo_irq'] = irq
                holder['mac_dl_bo_outliers'] = outliers
                    # =================================================
                holder['stream_sn2']=t['stream_sn2']
                holder['anomaly']=0
                data.append(holder)
        except KeyError:
            pass
    
        if len(data):
            return data[0]
        else:
            return -1
    #parse mac rlc size
    def parse_mac_rlc_size(self,json_object):
        self.axes[0,2].set_title('mac_rlc_size')
        df=pd.DataFrame()
        data =[]

        t= json_object
        try:
            if(t['stream_context_info']['hook_name']=='mac_rlc_size'):
                tmp= json.loads(t['stream_payload_msg'])
                holder={}
                batch=tmp['batch']
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                k=0
                holder['mac_rlc_size']=0
                for k in range(len(batch)):
                    holder['mac_rlc_size']+=batch[k]['size']
                holder['stream_sn']=t['stream_sn']
                holder['stream_sn2']=t['stream_sn2']
                holder['anomaly']=0
            #=============hisotgram calc==============================
                max = self.get_max(tmp['packetSizeMax'])
                min = self.get_min(tmp['packetSizeMin'])
                min_max_range = max - min
                hist__values = tmp['packetSizeHist']
                hist__key = 'packetSize'
                hist__shift = self.hist_shift['f1u_rlc_size']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['mac_rlc_size_max'] = max
                holder['mac_rlc_size_min'] = min
                holder['mac_rlc_size_range'] = min_max_range
                holder['mac_rlc_size_mean'] = mean
                holder['mac_rlc_size_var'] = var
                holder['mac_rlc_size_std'] = std
                holder['mac_rlc_size_skewness'] = skewness
                holder['mac_rlc_size_kurtosis'] = kurtosis
                holder['mac_rlc_size_irq'] = irq
                holder['mac_rlc_size_outliers'] = outliers
                    # =================================================

                data.append(holder)
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1

    # parse rlc f1u size
    def parse_rlc_f1u_size(self, json_object):
        self.axes[1,0].set_title('rlc_f1u_size')
        df=pd.DataFrame()

        data =[]
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']=='rlc_f1u_size'):
                tmp= json.loads(t['stream_payload_msg'])
                holder={}
                batch=tmp['batch']
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                k=0
                holder['rlc_f1u_size']=0
                for k in range(len(batch)):
                    holder['rlc_f1u_size']+=batch[k]['size']
                holder['anomaly']=0
                holder['stream_sn']=t['stream_sn']
                holder['stream_sn2']=t['stream_sn2']
        #=============hisotgram calc==============================
                max = self.get_max(tmp['packetSizeMax'])
                min = self.get_min(tmp['packetSizeMin'])
                min_max_range = max - min
                hist__values = tmp['packetSizeHist']
                hist__key = 'packetSize'
                hist__shift = self.hist_shift['rlc_f1u_size']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['rlc_f1u_size_max'] = max
                holder['rlc_f1u_size_min'] = min
                holder['rlc_f1u_size_range'] = min_max_range
                holder['rlc_f1u_size_mean'] = mean
                holder['rlc_f1u_size_var'] = var
                holder['rlc_f1u_size_std'] = std
                holder['rlc_f1u_size_skewness'] = skewness
                holder['rlc_f1u_size_kurtosis'] = kurtosis
                holder['rlc_f1u_size_irq'] = irq
                holder['rlc_f1u_size_outliers'] = outliers
                    # =================================================



                data.append(holder)
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1

 
    #parse rlc mac size
    def parse_rlc_mac_size(self,json_object):
        # self.axes[1,1].set_title('rlc_mac_size')
        df=pd.DataFrame()
        # f=open(self.filename)
        data =[]

        t=json_object
        try:
            if(t['stream_context_info']['hook_name']=='rlc_mac_size'):
                tmp= json.loads(t['stream_payload_msg'])
                holder={}
                batch=tmp['batch']
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                k=0
                holder['rlc_mac_size']=0
                for k in range(len(batch)):
                    holder['rlc_mac_size']+=batch[k]['size']
                holder['stream_sn']=t['stream_sn']
                holder['stream_sn2']=t['stream_sn2']
                holder['anomaly']=0
        #=============hisotgram calc==============================
                max = self.get_max(tmp['packetSizeMax'])
                min = self.get_min(tmp['packetSizeMin'])
                min_max_range = max - min
                hist__values = tmp['packetSizeHist']
                hist__key = 'packetSize'
                hist__shift = self.hist_shift['rlc_mac_size']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['rlc_mac_size_max'] = max
                holder['rlc_mac_size_min'] = min
                holder['rlc_mac_size_range'] = min_max_range
                holder['rlc_mac_size_mean'] = mean
                holder['rlc_mac_size_var'] = var
                holder['rlc_mac_size_std'] = std
                holder['rlc_mac_size_skewness'] = skewness
                holder['rlc_mac_size_kurtosis'] = kurtosis
                holder['rlc_mac_size_irq'] = irq
                holder['rlc_mac_size_outliers'] = outliers
                    # =================================================

                data.append(holder)
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1


    # parse mhtx
    def parse_mhtx(self, json_object):
        # self.axes[1,2].set_title('mhtx_agg')
        df=pd.DataFrame()
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['codeletSetId']==Hook_Vs_Codelet['midhaul_tx']):
                tmp= json.loads(t['stream_payload_msg'])
                holder={}
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                if('batchOut' in tmp.keys()):
            #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxOut'])
                    min = self.get_min(tmp['packetSizeMinOut'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistOut']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['mhtx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['MHTX_Out_max'] = max
                    holder['MHTX_Out_min'] = min
                    holder['MHTX_Out_range'] = min_max_range
                    holder['MHTX_Out_mean'] = mean
                    holder['MHTX_Out_var'] = var
                    holder['MHTX_Out_std'] = std
                    holder['MHTX_Out_skewness'] = skewness
                    holder['MHTX_Out_kurtosis'] = kurtosis
                    holder['MHTX_Out_irq'] = irq
                    holder['MHTX_Out_outliers'] = outliers
                    # =================================================

                    batch=tmp['batchOut']
                    k=0
                    holder['MHTX_Out_size']=0
                    for k in range(len(batch)):
                        holder['MHTX_Out_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)
                else:
            #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxIn'])
                    min = self.get_min(tmp['packetSizeMinIn'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistIn']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['mhtx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['MHTX_In_max'] = max
                    holder['MHTX_In_min'] = min
                    holder['MHTX_In_range'] = min_max_range
                    holder['MHTX_In_mean'] = mean
                    holder['MHTX_In_var'] = var
                    holder['MHTX_In_std'] = std
                    holder['MHTX_In_skewness'] = skewness
                    holder['MHTX_In_kurtosis'] = kurtosis
                    holder['MHTX_In_irq'] = irq
                    holder['MHTX_In_outliers'] = outliers
                    #============================================================

                    batch=tmp['batchIn']
                    k=0
                    holder['MHTX_In_size']=0
                    for k in range(len(batch)):
                        holder['MHTX_In_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1

    # parse bhtx
    def parse_bhtx(self, json_object):
        # self.axes[2,0].set_title('bhtx_ps')
        df=pd.DataFrame()
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['codeletSetId']==Hook_Vs_Codelet['backhaul_tx']):
                tmp= json.loads(t['stream_payload_msg'])
                # pprint.pprint (tmp)
                # break;
                holder={}
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                # pprint.pprint (tmp)
                if('batchOut' in tmp.keys()):
                #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxOut'])
                    min = self.get_min(tmp['packetSizeMinOut'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistOut']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['bhtx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['BHTX_Out_max'] = max
                    holder['BHTX_Out_min'] = min
                    holder['BHTX_Out_range'] = min_max_range
                    holder['BHTX_Out_mean'] = mean
                    holder['BHTX_Out_var'] = var
                    holder['BHTX_Out_std'] = std
                    holder['BHTX_Out_skewness'] = skewness
                    holder['BHTX_Out_kurtosis'] = kurtosis
                    holder['BHTX_Out_irq'] = irq
                    holder['BHTX_Out_outliers'] = outliers
                    #============================================================

                    # pprint.pprint(tmp)
                    # break;
                    batch=tmp['batchOut']
                    # k=0
                    holder['BHTX_Out_size']=0
                    holder['BHTX_Out_size'] += sum(entry['size'] for entry in batch)
                    # for k in range(len(batch)):
                    #     holder['out_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)
                else:
                    #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxIn'])
                    min = self.get_min(tmp['packetSizeMinIn'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistIn']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['bhtx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['BHTX_In_max'] = max
                    holder['BHTX_In_min'] = min
                    holder['BHTX_In_range'] = min_max_range
                    holder['BHTX_In_mean'] = mean
                    holder['BHTX_In_var'] = var
                    holder['BHTX_In_std'] = std
                    holder['BHTX_In_skewness'] = skewness
                    holder['BHTX_In_kurtosis'] = kurtosis
                    holder['BHTX_In_irq'] = irq
                    holder['BHTX_In_outliers'] = outliers
                    #============================================================
                    batch=tmp['batchIn']
                    # pprint.pprint(tmp)
                    # print("batch, len(batch):",batch, len(batch))
                    # k=0
                    holder['BHTX_In_size']=0
                    # print ("batch[0]['size']:",batch[0]['size'])
                    holder['BHTX_In_size'] += sum(entry['size'] for entry in batch)
                    # for k in range(len(batch)):
                    #     holder['in_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)

        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1

    # parse bhrx
    def parse_bhrx(self, json_object):
        # self.axes[2,1].set_title('bhrx_agg')
        df=pd.DataFrame()
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['codeletSetId']==Hook_Vs_Codelet['backhaul_rx']):
                tmp= json.loads(t['stream_payload_msg'])
                holder={}
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                if('batchOut' in tmp.keys()):
                #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxOut'])
                    min = self.get_min(tmp['packetSizeMinOut'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistOut']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['bhrx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['BHRX_Out_max'] = max
                    holder['BHRX_Out_min'] = min
                    holder['BHRX_Out_range'] = min_max_range
                    holder['BHRX_Out_mean'] = mean
                    holder['BHRX_Out_var'] = var
                    holder['BHRX_Out_std'] = std
                    holder['BHRX_Out_skewness'] = skewness
                    holder['BHRX_Out_kurtosis'] = kurtosis
                    holder['BHRX_Out_irq'] = irq
                    holder['BHRX_Out_outliers'] = outliers
                    #================================================================
                    batch=tmp['batchOut']
                    k=0
                    holder['BHRX_Out_size']=0
                    for k in range(len(batch)):
                        holder['BHRX_Out_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)
                else:
                    #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxIn'])
                    min = self.get_min(tmp['packetSizeMinIn'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistIn']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['bhrx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['BHRX_In_max'] = max
                    holder['BHRX_In_min'] = min
                    holder['BHRX_In_range'] = min_max_range
                    holder['BHRX_In_mean'] = mean
                    holder['BHRX_In_var'] = var
                    holder['BHRX_In_std'] = std
                    holder['BHRX_In_skewness'] = skewness
                    holder['BHRX_In_kurtosis'] = kurtosis
                    holder['BHRX_In_irq'] = irq
                    holder['BHRX_In_outliers'] = outliers
                    #============================================================

                    batch=tmp['batchIn']
                    k=0
                    holder['BHRX_In_size']=0
                    for k in range(len(batch)):
                        holder['BHRX_In_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)

        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1



    # parse mhrx
    def parse_mhrx(self, json_object):
        # self.axes[2,2].set_title('mhrx_agg')
        df=pd.DataFrame()
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['codeletSetId']==Hook_Vs_Codelet['midhaul_rx']):
                tmp= json.loads(t['stream_payload_msg'])
                holder={}
                holder['timestamp']=tmp['timestamp']
                holder['cellId']=tmp['cellId']
                if('batchOut' in tmp.keys()):
                #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxOut'])
                    min = self.get_min(tmp['packetSizeMinOut'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistOut']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['mhrx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['MHRX_Out_max'] = max
                    holder['MHRX_Out_min'] = min
                    holder['MHRX_Out_range'] = min_max_range
                    holder['MHRX_Out_mean'] = mean
                    holder['MHRX_Out_var'] = var
                    holder['MHRX_Out_std'] = std
                    holder['MHRX_Out_skewness'] = skewness
                    holder['MHRX_Out_kurtosis'] = kurtosis
                    holder['MHRX_Out_irq'] = irq
                    holder['MHRX_Out_outliers'] = outliers
                            # =================================================

                    batch=tmp['batchOut']
                    k=0
                    holder['MHRX_Out_size']=0
                    for k in range(len(batch)):
                        holder['MHRX_Out_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)
                else:
                    #=============hisotgram calc==============================
                    max = self.get_max(tmp['packetSizeMaxIn'])
                    min = self.get_min(tmp['packetSizeMinIn'])
                    min_max_range = max - min
                    hist__values = tmp['packetSizeHistIn']
                    hist__key = 'packetSize'
                    hist__shift = self.hist_shift['mhrx_ps']
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                    holder['MHRX_In_max'] = max
                    holder['MHRX_In_min'] = min
                    holder['MHRX_In_range'] = min_max_range
                    holder['MHRX_In_mean'] = mean
                    holder['MHRX_In_var'] = var
                    holder['MHRX_In_std'] = std
                    holder['MHRX_In_skewness'] = skewness
                    holder['MHRX_In_kurtosis'] = kurtosis
                    holder['MHRX_In_irq'] = irq
                    holder['MHRX_In_outliers'] = outliers
                    #============================================================

                    batch=tmp['batchIn']
                    k=0
                    holder['MHRX_In_size']=0
                    for k in range(len(batch)):
                        holder['MHRX_In_size']+=batch[k]['size']
                    holder['stream_sn']=t['stream_sn']
                    holder['stream_sn2']=t['stream_sn2']
                    holder['anomaly']=0
                    data.append(holder)

        except KeyError:
            pass

        len(data)
        if len(data):
            return data[0]
        else:
            return -1



    # parse fapi dl config
    def parse_fapi_dl_config(self, json_object):
        df=pd.DataFrame()
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']=='fapi_gnb_dl_config_req'):
                tmp=json.loads(t['stream_payload_msg'])
                # pprint.pprint(tmp)
                # break;
                avg_l1_prbSize = self.avg_packet_calc_hist(tmp['l1DlcPrbHist'], 'prbSize', self.hist_shift['fapi_gnb_dl_config_req_prb'], self.get_min(tmp['l1PrbMin']), self.get_max(tmp['l1PrbMax']))
                # print(avg_l1_prbSize)
                avg_l1_mcs = self.avg_packet_calc_hist(tmp['l1DlcMcsHist'], 'mcs', self.hist_shift['fapi_gnb_dl_config_req_mcs'], self.get_min(tmp['l1McsMin']), self.get_max(tmp['l1McsMax']))
                # print(avg_l1_mcs)
                avg_l1_tbs = self.avg_packet_calc_hist(tmp['l1DlcTbsHist'], 'tbsSize', self.hist_shift['fapi_gnb_dl_config_req_tbs'], self.get_min(tmp['l1TbsMin']), self.get_max(tmp['l1TbsMax']))
                # print(avg_l1_tbs)
                pdsch_cnt = self.cnt_calculator(tmp['l1DlcTbsHist'], 'rnti', 'tbsSize', 'cnt')
                # print(pdsch_cnt)
                avg_pdsch = pdsch_cnt / len(tmp['l1DlcTbsHist'])

                # avg_pdsch = np.mean(list(self.avg_packet_calc(tmp['l1DlcTbsHist'], 'rnti', 'tbsSize', 'cnt').values()))

                # l1_tx = tmp['l1DlcTx'][0]['tx']
                # sum traffic for all rnti
                sum_tx = 0
                for item in tmp['l1DlcTx']:
                    sum_tx += item['tx']
                l1_tx = sum_tx

                tdf=pd.DataFrame()
                # print ('timestamp','avg_l1_prbSize', avg_l1_prbSize,'avg_l1_mcs', avg_l1_mcs,'avg_l1_tbs', avg_l1_tbs,'l1_tx', l1_tx,'l1McsMin', self.get_min(tmp['l1McsMin']),'l1McsMax', self.get_max(tmp['l1McsMax']),'l1TbsMax', self.get_max(tmp['l1TbsMax']),'l1TbsMin', self.get_min(tmp['l1TbsMin']),'l1PrbMax', self.get_max(tmp['l1PrbMax']),'l1PrbMin', self.get_min(tmp['l1PrbMin']),'pdsch_count':pdsch_cnt, 'avg_pdsch', avg_pdsch ,'anomaly', 0)
                # tdf=pd.DataFrame({'timestamp': tmp['timestamp'],'avg_l1_prbSize': avg_l1_prbSize,'avg_l1_mcs': avg_l1_mcs,'avg_l1_tbs': avg_l1_tbs,'l1_tx': l1_tx },index=[0])
                data.append({'timestamp': tmp['timestamp'],'l1_avg_prbSize': avg_l1_prbSize,'l1_avg_mcs': avg_l1_mcs,'l1_avg_tbs': avg_l1_tbs,'l1_tx': l1_tx,'l1_McsMin':self.get_min(tmp['l1McsMin']),'l1_McsMax':self.get_max(tmp['l1McsMax']),'l1_TbsMax':self.get_max(tmp['l1TbsMax']),'l1_TbsMin':self.get_min(tmp['l1TbsMin']),'l1_PrbMax':self.get_max(tmp['l1PrbMax']),'l1_PrbMin':self.get_min(tmp['l1PrbMin']),'l1_pdsch_count':pdsch_cnt, 'l1_avg_pdsch':avg_pdsch ,'anomaly':0})
        except KeyError:
            pass
 
    # pprint.pprint(data)
        if len(data):
            return data[0]
        else:
            return -1



    # parse mac bsr update
    def parse_mac_bsr_update(self, json_object):
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']== 'mac_bsr_update'):

                tmp=json.loads(t['stream_payload_msg'])
                holder={}
                holder['timestamp']=tmp['timestamp']
        #=============hisotgram calc==============================
                max = self.get_max(tmp['l2BsrMax'])
                min = self.get_min(tmp['l2BsrMin'])
                min_max_range = max - min
                hist__values = tmp['l2BsrHist']
                hist__key = 'queueLoad'
                hist__shift = self.hist_shift['mac_bsr_update']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['mac_bsr_update_max'] = max
                holder['mac_bsr_update_min'] = min
                holder['mac_bsr_update_range'] = min_max_range
                holder['mac_bsr_update_mean'] = mean
                holder['mac_bsr_update_var'] = var
                holder['mac_bsr_update_std'] = std
                holder['mac_bsr_update_skewness'] = skewness
                holder['mac_bsr_update_kurtosis'] = kurtosis
                holder['mac_bsr_update_irq'] = irq
                holder['mac_bsr_update_outliers'] = outliers
                holder['anomaly']=0
                data.append(holder)
                    # =================================================
                # data.append(holder) #,'l2BsrMin':tmp['l2BsrMin'][0]['val']
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1


    # parse mac csi update
    def parse_mac_csi_report(self, json_object):
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']== 'mac_csi_report'):
                tmp=json.loads(t['stream_payload_msg'])
                holder={}
                holder['timestamp']=tmp['timestamp']
                # pprint.pprint(tmp)
                # break;
        #=============hisotgram calc==============================
                max = self.get_max(tmp['l2CsiMax'])
                min = self.get_min(tmp['l2CsiMin'])
                min_max_range = max - min
                hist__values = tmp['l2CsiHist']
                hist__key = 'val'
                hist__shift = self.hist_shift['mac_csi_report']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['mac_csi_report_max'] = max
                holder['mac_csi_report_min'] = min
                holder['mac_csi_report_range'] = min_max_range
                holder['mac_csi_report_mean'] = mean
                holder['mac_csi_report_var'] = var
                holder['mac_csi_report_std'] = std
                holder['mac_csi_report_skewness'] = skewness
                holder['mac_csi_report_kurtosis'] = kurtosis
                holder['mac_csi_report_irq'] = irq
                holder['mac_csi_report_outliers'] = outliers
                holder['anomaly']=0
                data.append(holder)
        # =================================================
                # avg_l2_Csi = math.ceil(list(self.avg_calculator(tmp['l2CsiHist'], 'rnti', 'val', 'cnt').values())[0])
                # data.append({'timestamp': tmp['timestamp'],'l2CsiMin':self.get_min(tmp['l2CsiMin']),'l2CsiMax':self.get_max(tmp['l2CsiMax']),'anomaly':0}) #,'l2CsiMin':tmp['l2CsiMin'][0]['val']

                # break;
        except KeyError:
            pass
        len(data)
        if len(data):
            return data[0]
        else:
            return -1

        df = pd.DataFrame.from_dict(data)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ns')
        df=df.sort_values(by='timestamp', ascending=True)
        df.set_index('timestamp', inplace=True)
        return df

    # parse mac dl harq
    def parse_mac_dl_harq(self, json_object):
        data =[]
        # f=open(self.filename)
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']== 'mac_dl_harq'):
                tmp=json.loads(t['stream_payload_msg'])
                # pprint.pprint(tmp)
                # break;
                total = 0
                nack = 0
                ack = 0
                for  i in tmp['l2HarqCnt']:
                    total+=i['cnt']
                    nack+=i['nack']
                    ack+=i['ack']
                consMax = 0
                for i in tmp['l2HarqConsMax']:
                    consMax = np.max(i['val'])
                # tdf=pd.DataFrame({'timestamp': tmp['timestamp'],'avg_l2_Harq_ACK':avg_l2_Harq_ACK, 'avg_l2_Harq_NACK':avg_l2_Harq_NACK, 'avg_l2_Harq_DTX':avg_l2_Harq_DTX },index=[0])
                data.append({'timestamp': tmp['timestamp'],'mac_dl_harq_total': total, 'mac_dl_harq_NACK': nack, 'mac_dl_harq_ACK': ack,'mac_dl_harq_max_cons_loss':consMax, 'mac_dl_harq_nack_rate' : (nack/total) , 'anomaly':0 }) # ,'l2HarqConsMax':tmp['l2HarqConsMax'][0]['val']

                # break;
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1

        df = pd.DataFrame.from_dict(data)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ns')
        df=df.sort_values(by='timestamp', ascending=True)
        df.set_index('timestamp', inplace=True)
        df
        for i in range(len(self.timestamp)):
            t=df.between_time(self.timestamp.loc[i]['start_time'].time(), self.timestamp.loc[i]['end_time'].time())['anomaly']
            for j in range(len(t.index)):
                df.at[t.index[j],'anomaly']=1
        return df

    # parse mac sinr update
    def parse_mac_sinr_update(self, json_object):
        data =[]
        # f=open(self.filename)
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']== 'mac_sinr_update'):
                tmp=json.loads(t['stream_payload_msg'])
                holder={}
                holder['timestamp']=tmp['timestamp']
        #=============hisotgram calc==============================
                max = self.get_max(tmp['l2SinrMax'])
                min = self.get_min(tmp['l2SinrMin'])
                min_max_range = max - min
                hist__values = tmp['l2SinrHist']
                hist__key = 'val'
                hist__shift = self.hist_shift['mac_sinr_update']
                mean, var ,std, skewness, kurtosis, irq,  outliers = self.process_packet_hist(hist__values, hist__key, hist__shift, min, max)
                holder['mac_sinr_update_max'] = max
                holder['mac_sinr_update_min'] = min
                holder['mac_sinr_update_range'] = min_max_range
                holder['mac_sinr_update_mean'] = mean
                holder['mac_sinr_update_var'] = var
                holder['mac_sinr_update_std'] = std
                holder['mac_sinr_update_skewness'] = skewness
                holder['mac_sinr_update_kurtosis'] = kurtosis
                holder['mac_sinr_update_irq'] = irq
                holder['mac_sinr_update_outliers'] = outliers
                holder['anomaly']=0
                data.append(holder)
        # =================================================


                # avg_l2_Sinr = math.ceil(list(self.avg_calculator(tmp['l2SinrHist'], 'rnti', 'val', 'cnt').values())[0])
                # data.append({'timestamp': tmp['timestamp'],'l2SinrMax':self.get_max(tmp['l2SinrMax']),'l2SinrMin':self.get_min(tmp['l2SinrMin']), 'anomaly':0})
        except KeyError:
            pass
    # f.close()
        if len(data):
            return data[0]
        else:
            return -1


    # parse mac ul crc ind
    def parse_mac_ul_crc_ind(self, json_object):
        data =[]
        # f=open(self.filename)
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']== 'mac_ul_crc_ind'):
                tmp=json.loads(t['stream_payload_msg'])
                # print (tmp)
                # break;

                # avg_l2_Crc = math.ceil(list(self.avg_calculator(tmp['l2CrcHist'], 'rnti', 'loss', 'cnt').values())[0])
                loss = 0
                total = 0
                for i in tmp['l2Avg']:
                    loss+=i['loss']
                    total+=i['total']

                data.append({'timestamp': tmp['timestamp'],'mac_ul_CRC_Max':self.get_max(tmp['l2Max']),'mac_ul_CRC_Min':self.get_min(tmp['l2Max']), 'mac_ul_CRC_Loss': loss,'mac_ul_CRC_Loss_Rate': loss/total, 'mac_ul_CRC_Total':total, 'anomaly': 0 }) # ,'l2CrcMax':tmp['l2CrcMax'][0]['val']

                # break;
        except KeyError:
            pass
        if len(data):
            return data[0]
        else:
            return -1




    # parse fapi gnb ul config req
    def parse_fapi_gnb_ul_config_req(self, json_object):
        data = []
        # f=open(self.filename)
        # for i in f:
        t=json_object
        try:
            if(t['stream_context_info']['hook_name']== 'fapi_gnb_ul_config_req'):
                tmp=json.loads(t['stream_payload_msg'])
                # pprint.pprint(tmp)
                # break;
                # pucsh_count = 0
                # for i in tmp['l1UlcTbsHist']:
                #     pucsh_count+=i['cnt']
                # avg_pusch = np.mean(list(self.avg_packet_calc(tmp['l1UlcPrbHist'], 'prbSize', 'tbsSize', 'cnt').values()))
                # avg_l1_UL_prbSize = math.ceil(list(self.avg_calculator(tmp['l1UlcPrbHist'], 'rnti', 'prbSize', 'cnt').values())[0])
                # avg_l1_UL_mcs = math.ceil(list(self.avg_calculator(tmp['l1UlcMcsHist'], 'rnti', 'mcs', 'cnt').values())[0])
                # avg_l1_UL_tbs = math.ceil(list(self.avg_calculator(tmp['l1UlcTbsHist'], 'rnti', 'tbsSize', 'cnt').values())[0])

                avg_l1_UL_prbSize = self.avg_packet_calc_hist(tmp['l1UlcPrbHist'], 'prbSize', self.hist_shift['fapi_gnb_ul_config_req_prb'], self.get_min(tmp['l1PrbMin']), self.get_max(tmp['l1PrbMax']))
                avg_l1_UL_mcs = self.avg_packet_calc_hist(tmp['l1UlcMcsHist'], 'mcs', self.hist_shift['fapi_gnb_ul_config_req_mcs'], self.get_min(tmp['l1McsMin']), self.get_max(tmp['l1McsMax']))
                avg_l1_UL_tbs = self.avg_packet_calc_hist(tmp['l1UlcTbsHist'], 'tbsSize', self.hist_shift['fapi_gnb_ul_config_req_tbs'], self.get_min(tmp['l1TbsMin']), self.get_max(tmp['l1TbsMax']))
                pucsh_cnt = self.cnt_calculator(tmp['l1UlcTbsHist'], 'rnti', 'tbsSize', 'cnt')
                avg_pusch = pucsh_cnt / len(tmp['l1UlcTbsHist'])


                l1_UL_tx=0
                for i in tmp['l1UlcTx']:
                    l1_UL_tx += i['tx']
                # tdf=pd.DataFrame({'timestamp': tmp['timestamp'],'avg_l1_UL_prbSize': avg_l1_UL_prbSize,'avg_l1_UL_mcs': avg_l1_UL_mcs,'avg_l1_UL_tbs': avg_l1_UL_tbs,'l1_UL_tx': l1_UL_tx },index=[0])
                data.append({'timestamp': tmp['timestamp'], 'l1_pusch_count': pucsh_cnt, 'l1_avg_pusch_count': avg_pusch ,'l1_avg_UL_prbSize': avg_l1_UL_prbSize,'l1_avg_UL_mcs': avg_l1_UL_mcs,'l1_avg_UL_tbs': avg_l1_UL_tbs,'l1_UL_tx': l1_UL_tx,'l1_ULMcsMin':self.get_min(tmp['l1McsMin']),'l1_ULMcsMax':self.get_min(tmp['l1McsMax']),'l1_ULTbsMax':self.get_min(tmp['l1TbsMax']),'l1_ULTbsMin':self.get_min(tmp['l1TbsMin']),'l1_ULPrbMax':self.get_max(tmp['l1PrbMax']),'l1_ULPrbMin':self.get_max(tmp['l1PrbMin']), 'anomaly':0  })

                # break;
        except KeyError:
            pass
        if len(data):
            return data[0]
        else :
            return -1
        df = pd.DataFrame.from_dict(data)

        df['timestamp'] = df['timestamp'].astype(np.int64)
        df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ns')
        df=df.sort_values(by='timestamp', ascending=True)
        df.set_index('timestamp', inplace=True)
        df
        for i in range(len(self.timestamp)):
            t=df.between_time(self.timestamp.loc[i]['start_time'].time(), self.timestamp.loc[i]['end_time'].time())['anomaly']
            for j in range(len(t.index)):
                df.at[t.index[j],'anomaly']=1
        return df


    # parse platform kpi
    def parse_platform(self, json_object):
        # f=open(self.filename)
        data =[]
        # for i in f:
        t=json_object
        # pprint.pprint(t['processes'])
        # pprint.pprint(t['stream_type'])
        # pprint.pprint(t['host_name'])
        try:
            if(t['stream_type']=='ebpf' and t['host_name']=='telco-2'):
                tmp = t['processes']
                # pprint.pprint("1")
                total_bin = [0]*16
                max_runtime = 0
                for i in tmp:
                    if(i['process_name'] != 'OS swapper'):
                        for j in i['threads']:
                            total_bin = [x + y for x, y in zip(total_bin, j['bins'])]
                            max_runtime = max(max_runtime,j['max'])
                # pprint.pprint("2")
                for i in tmp:
                    for j in i['threads']:
                        holder = {}
                        holder['process_name'] = i['process_name']
                        holder['process_id'] = i['process_id']
                        holder['thread_id']=j['thread_id']
                        holder['thread_name']=j['thread_name']
                        holder['max']=j['max']
                        holder['total_events']=j['total']

                        holder['others_runtime_bin']= [a_i - b_i for a_i, b_i in zip(total_bin, j['bins'])]
                        holder['max_runtime'] = max_runtime
                        holder['others_runtime'] = self.calc_runtime(holder['others_runtime_bin'])

                        holder['total_runtime'] = self.calc_runtime(j['bins'])
                        holder['stream_id'] = t['stream_id']
                        holder['host_name'] = t['host_name']
                        holder['timestamp'] = t['stream_payload_time']
                        holder['cpu_id'] = t['cpu_entry']
                        holder['stream_sn'] = t['stream_sn']
                        holder['bins'] = j['bins']
                        holder['anomaly'] = 0
                        
                        data.append(holder)
        except KeyError:
            pass
        len(data)
        if len(data):
            return data[0]
        else :
            return -1
        df = pd.DataFrame.from_dict(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
        # correct_time = df['timestamp'].dt.time
        # desired_date = ts_df['start_time'][0].date()
        # corrected_timestamp = pd.to_datetime(correct_time.astype(str)).apply(
        #     lambda x: pd.Timestamp.combine(desired_date, x.time())
        # )
        # df['timestamp'] = corrected_timestamp


        # parse swtich kpi
        Ethernet_Port_df=pd.DataFrame()
    def parse_swtich(self, json_object):
        data =[]
        t=json_object
        try:

            if(t['host_name']=='5gmgmt'):

                tmp=json.loads(t['stream_payload_msg'])

                if(tmp['name']=='Ethernet31/1' or tmp['name']=='Ethernet15/2'):  #31/1 RU port, 15/2 DU Port

                    holder={}

                    holder['timestamp']=t['stream_payload_time']

                    holder['stream_sn']=t['stream_sn']

                    holder['name']=tmp['name']

                    holder['anomaly']=0

                    holder['in_octet_total'] = tmp['in-octets']

                    holder['out_octet_total'] = tmp['out-octets']

                    data.append(holder)

        except KeyError:
            pass
        len(data)
        if len(data):
            return data[0]
        else :
            return -1

    def merge_platform(self, All_Platform_KPIs_df, Ethernet_Port_pivot_df):
        # print(All_Platform_KPIs_df.columns)
        # print(Ethernet_Port_pivot_df.columns)
        All_Platform_KPIs_df['timestamp'] = All_Platform_KPIs_df['timestamp'].apply(lambda x: x.replace(year=Ethernet_Port_pivot_df.loc[0, 'timestamp'].year, month=Ethernet_Port_pivot_df.loc[0, 'timestamp'].month, day=Ethernet_Port_pivot_df.loc[0, 'timestamp'].day))

        All_Platform_KPIs_and_Ethernet_Port_df = pd.merge_asof(left=All_Platform_KPIs_df, right=Ethernet_Port_pivot_df, on='timestamp',allow_exact_matches=True, direction="forward", tolerance=pd.Timedelta("1150 milliseconds"))

        # imputing:
        All_Platform_KPIs_and_Ethernet_Port_df['in_octet_total_Ethernet15/2'] = All_Platform_KPIs_and_Ethernet_Port_df['in_octet_total_Ethernet15/2'].interpolate(method='linear')
        All_Platform_KPIs_and_Ethernet_Port_df['out_octet_total_Ethernet15/2'] = All_Platform_KPIs_and_Ethernet_Port_df['out_octet_total_Ethernet15/2'].interpolate(method='linear')
        All_Platform_KPIs_and_Ethernet_Port_df['in_octet_total_Ethernet31/1'] = All_Platform_KPIs_and_Ethernet_Port_df['in_octet_total_Ethernet31/1'].interpolate(method='linear')
        All_Platform_KPIs_and_Ethernet_Port_df['out_octet_total_Ethernet31/1'] = All_Platform_KPIs_and_Ethernet_Port_df['out_octet_total_Ethernet31/1'].interpolate(method='linear')


        All_Platform_KPIs_and_Ethernet_Port_df.fillna(0, inplace=True)


        #map platfrom KPIs to fix template of columns:
        # using col name filtered by chaunhao
        col_names = 'timestamp,dumgr_du_1_DUMGR_LOGGER_1_max,dumgr_du_1_dumgr_du_1_max,gnb_cu_l3_l3_main_max,gnb_cu_pdcp_0_0_f1_worker_0_max,gnb_cu_pdcp_0_0_pdcp_master_0_max,gnb_cu_pdcp_0_0_pdcp_worker_0_max,gnb_cu_pdcp_0_0_recv_data_0_max,gnb_du_layer2_LowPrio_DU1_C0_max,gnb_du_layer2_TxCtrl_DU1_C0_max,gnb_du_layer2_f1_du_worker_0_max,gnb_du_layer2_pr_accumulator_max,gnb_du_layer2_rlcAccum_DU1_max,gnb_du_layer2_rlcTimer_DU1_max,gnb_du_layer2_rlcWorkrDU1__max,l1app_main_ebbupool_act_0_max,l1app_main_ebbupool_act_1_max,l1app_main_ebbupool_act_2_max,l1app_main_ebbupool_act_3_max,l1app_main_fh_main_poll-22_max,l1app_main_fh_rx_bbdev-21_max,phc2sys_phc2sys_max,ptp4l_ptp4l_max,dumgr_du_1_DUMGR_LOGGER_1_total_events,dumgr_du_1_dumgr_du_1_total_events,gnb_cu_l3_l3_main_total_events,gnb_cu_pdcp_0_0_f1_worker_0_total_events,gnb_cu_pdcp_0_0_pdcp_master_0_total_events,gnb_cu_pdcp_0_0_pdcp_worker_0_total_events,gnb_cu_pdcp_0_0_recv_data_0_total_events,gnb_du_layer2_LowPrio_DU1_C0_total_events,gnb_du_layer2_TxCtrl_DU1_C0_total_events,gnb_du_layer2_f1_du_worker_0_total_events,gnb_du_layer2_pr_accumulator_total_events,gnb_du_layer2_rlcAccum_DU1_total_events,gnb_du_layer2_rlcTimer_DU1_total_events,gnb_du_layer2_rlcWorkrDU1__total_events,l1app_main_ebbupool_act_0_total_events,l1app_main_ebbupool_act_1_total_events,l1app_main_ebbupool_act_2_total_events,l1app_main_ebbupool_act_3_total_events,l1app_main_fh_main_poll-22_total_events,l1app_main_fh_rx_bbdev-21_total_events,phc2sys_phc2sys_total_events,ptp4l_ptp4l_total_events,dumgr_du_1_DUMGR_LOGGER_1_total_runtime,dumgr_du_1_dumgr_du_1_total_runtime,gnb_cu_l3_l3_main_total_runtime,gnb_cu_pdcp_0_0_f1_worker_0_total_runtime,gnb_cu_pdcp_0_0_pdcp_master_0_total_runtime,gnb_cu_pdcp_0_0_pdcp_worker_0_total_runtime,gnb_cu_pdcp_0_0_recv_data_0_total_runtime,gnb_du_layer2_LowPrio_DU1_C0_total_runtime,gnb_du_layer2_TxCtrl_DU1_C0_total_runtime,gnb_du_layer2_f1_du_worker_0_total_runtime,gnb_du_layer2_pr_accumulator_total_runtime,gnb_du_layer2_rlcAccum_DU1_total_runtime,gnb_du_layer2_rlcTimer_DU1_total_runtime,gnb_du_layer2_rlcWorkrDU1__total_runtime,l1app_main_ebbupool_act_0_total_runtime,l1app_main_ebbupool_act_1_total_runtime,l1app_main_ebbupool_act_2_total_runtime,l1app_main_ebbupool_act_3_total_runtime,l1app_main_fh_main_poll-22_total_runtime,l1app_main_fh_rx_bbdev-21_total_runtime,phc2sys_phc2sys_total_runtime,ptp4l_ptp4l_total_runtime,dumgr_du_1_DUMGR_LOGGER_1_mean,dumgr_du_1_dumgr_du_1_mean,gnb_cu_l3_l3_main_mean,gnb_cu_pdcp_0_0_f1_worker_0_mean,gnb_cu_pdcp_0_0_pdcp_master_0_mean,gnb_cu_pdcp_0_0_pdcp_worker_0_mean,gnb_cu_pdcp_0_0_recv_data_0_mean,gnb_du_layer2_LowPrio_DU1_C0_mean,gnb_du_layer2_TxCtrl_DU1_C0_mean,gnb_du_layer2_f1_du_worker_0_mean,gnb_du_layer2_pr_accumulator_mean,gnb_du_layer2_rlcAccum_DU1_mean,gnb_du_layer2_rlcTimer_DU1_mean,gnb_du_layer2_rlcWorkrDU1__mean,l1app_main_ebbupool_act_0_mean,l1app_main_ebbupool_act_1_mean,l1app_main_ebbupool_act_2_mean,l1app_main_ebbupool_act_3_mean,l1app_main_fh_main_poll-22_mean,l1app_main_fh_rx_bbdev-21_mean,phc2sys_phc2sys_mean,ptp4l_ptp4l_mean,dumgr_du_1_DUMGR_LOGGER_1_range,dumgr_du_1_dumgr_du_1_range,gnb_cu_l3_l3_main_range,gnb_cu_pdcp_0_0_f1_worker_0_range,gnb_cu_pdcp_0_0_pdcp_master_0_range,gnb_cu_pdcp_0_0_pdcp_worker_0_range,gnb_cu_pdcp_0_0_recv_data_0_range,gnb_du_layer2_LowPrio_DU1_C0_range,gnb_du_layer2_TxCtrl_DU1_C0_range,gnb_du_layer2_f1_du_worker_0_range,gnb_du_layer2_pr_accumulator_range,gnb_du_layer2_rlcAccum_DU1_range,gnb_du_layer2_rlcTimer_DU1_range,gnb_du_layer2_rlcWorkrDU1__range,l1app_main_ebbupool_act_0_range,l1app_main_ebbupool_act_1_range,l1app_main_ebbupool_act_2_range,l1app_main_ebbupool_act_3_range,l1app_main_fh_main_poll-22_range,l1app_main_fh_rx_bbdev-21_range,phc2sys_phc2sys_range,ptp4l_ptp4l_range,dumgr_du_1_DUMGR_LOGGER_1_var,dumgr_du_1_dumgr_du_1_var,gnb_cu_l3_l3_main_var,gnb_cu_pdcp_0_0_f1_worker_0_var,gnb_cu_pdcp_0_0_pdcp_master_0_var,gnb_cu_pdcp_0_0_pdcp_worker_0_var,gnb_cu_pdcp_0_0_recv_data_0_var,gnb_du_layer2_LowPrio_DU1_C0_var,gnb_du_layer2_TxCtrl_DU1_C0_var,gnb_du_layer2_f1_du_worker_0_var,gnb_du_layer2_pr_accumulator_var,gnb_du_layer2_rlcAccum_DU1_var,gnb_du_layer2_rlcTimer_DU1_var,gnb_du_layer2_rlcWorkrDU1__var,l1app_main_ebbupool_act_0_var,l1app_main_ebbupool_act_1_var,l1app_main_ebbupool_act_2_var,l1app_main_ebbupool_act_3_var,l1app_main_fh_main_poll-22_var,l1app_main_fh_rx_bbdev-21_var,phc2sys_phc2sys_var,ptp4l_ptp4l_var,dumgr_du_1_DUMGR_LOGGER_1_std,dumgr_du_1_dumgr_du_1_std,gnb_cu_l3_l3_main_std,gnb_cu_pdcp_0_0_f1_worker_0_std,gnb_cu_pdcp_0_0_pdcp_master_0_std,gnb_cu_pdcp_0_0_pdcp_worker_0_std,gnb_cu_pdcp_0_0_recv_data_0_std,gnb_du_layer2_LowPrio_DU1_C0_std,gnb_du_layer2_TxCtrl_DU1_C0_std,gnb_du_layer2_f1_du_worker_0_std,gnb_du_layer2_pr_accumulator_std,gnb_du_layer2_rlcAccum_DU1_std,gnb_du_layer2_rlcTimer_DU1_std,gnb_du_layer2_rlcWorkrDU1__std,l1app_main_ebbupool_act_0_std,l1app_main_ebbupool_act_1_std,l1app_main_ebbupool_act_2_std,l1app_main_ebbupool_act_3_std,l1app_main_fh_main_poll-22_std,l1app_main_fh_rx_bbdev-21_std,phc2sys_phc2sys_std,ptp4l_ptp4l_std,dumgr_du_1_DUMGR_LOGGER_1_skewness,dumgr_du_1_dumgr_du_1_skewness,gnb_cu_l3_l3_main_skewness,gnb_cu_pdcp_0_0_f1_worker_0_skewness,gnb_cu_pdcp_0_0_pdcp_master_0_skewness,gnb_cu_pdcp_0_0_pdcp_worker_0_skewness,gnb_cu_pdcp_0_0_recv_data_0_skewness,gnb_du_layer2_LowPrio_DU1_C0_skewness,gnb_du_layer2_TxCtrl_DU1_C0_skewness,gnb_du_layer2_f1_du_worker_0_skewness,gnb_du_layer2_pr_accumulator_skewness,gnb_du_layer2_rlcAccum_DU1_skewness,gnb_du_layer2_rlcTimer_DU1_skewness,gnb_du_layer2_rlcWorkrDU1__skewness,l1app_main_ebbupool_act_0_skewness,l1app_main_ebbupool_act_1_skewness,l1app_main_ebbupool_act_2_skewness,l1app_main_ebbupool_act_3_skewness,l1app_main_fh_main_poll-22_skewness,l1app_main_fh_rx_bbdev-21_skewness,phc2sys_phc2sys_skewness,ptp4l_ptp4l_skewness,dumgr_du_1_DUMGR_LOGGER_1_kurtosis,dumgr_du_1_dumgr_du_1_kurtosis,gnb_cu_l3_l3_main_kurtosis,gnb_cu_pdcp_0_0_f1_worker_0_kurtosis,gnb_cu_pdcp_0_0_pdcp_master_0_kurtosis,gnb_cu_pdcp_0_0_pdcp_worker_0_kurtosis,gnb_cu_pdcp_0_0_recv_data_0_kurtosis,gnb_du_layer2_LowPrio_DU1_C0_kurtosis,gnb_du_layer2_TxCtrl_DU1_C0_kurtosis,gnb_du_layer2_f1_du_worker_0_kurtosis,gnb_du_layer2_pr_accumulator_kurtosis,gnb_du_layer2_rlcAccum_DU1_kurtosis,gnb_du_layer2_rlcTimer_DU1_kurtosis,gnb_du_layer2_rlcWorkrDU1__kurtosis,l1app_main_ebbupool_act_0_kurtosis,l1app_main_ebbupool_act_1_kurtosis,l1app_main_ebbupool_act_2_kurtosis,l1app_main_ebbupool_act_3_kurtosis,l1app_main_fh_main_poll-22_kurtosis,l1app_main_fh_rx_bbdev-21_kurtosis,phc2sys_phc2sys_kurtosis,ptp4l_ptp4l_kurtosis,dumgr_du_1_DUMGR_LOGGER_1_irq,dumgr_du_1_dumgr_du_1_irq,gnb_cu_l3_l3_main_irq,gnb_cu_pdcp_0_0_f1_worker_0_irq,gnb_cu_pdcp_0_0_pdcp_master_0_irq,gnb_cu_pdcp_0_0_pdcp_worker_0_irq,gnb_cu_pdcp_0_0_recv_data_0_irq,gnb_du_layer2_LowPrio_DU1_C0_irq,gnb_du_layer2_TxCtrl_DU1_C0_irq,gnb_du_layer2_f1_du_worker_0_irq,gnb_du_layer2_pr_accumulator_irq,gnb_du_layer2_rlcAccum_DU1_irq,gnb_du_layer2_rlcTimer_DU1_irq,gnb_du_layer2_rlcWorkrDU1__irq,l1app_main_ebbupool_act_0_irq,l1app_main_ebbupool_act_1_irq,l1app_main_ebbupool_act_2_irq,l1app_main_ebbupool_act_3_irq,l1app_main_fh_main_poll-22_irq,l1app_main_fh_rx_bbdev-21_irq,phc2sys_phc2sys_irq,ptp4l_ptp4l_irq,dumgr_du_1_DUMGR_LOGGER_1_outliers,dumgr_du_1_dumgr_du_1_outliers,gnb_cu_l3_l3_main_outliers,gnb_cu_pdcp_0_0_f1_worker_0_outliers,gnb_cu_pdcp_0_0_pdcp_master_0_outliers,gnb_cu_pdcp_0_0_pdcp_worker_0_outliers,gnb_cu_pdcp_0_0_recv_data_0_outliers,gnb_du_layer2_LowPrio_DU1_C0_outliers,gnb_du_layer2_TxCtrl_DU1_C0_outliers,gnb_du_layer2_f1_du_worker_0_outliers,gnb_du_layer2_pr_accumulator_outliers,gnb_du_layer2_rlcAccum_DU1_outliers,gnb_du_layer2_rlcTimer_DU1_outliers,gnb_du_layer2_rlcWorkrDU1__outliers,l1app_main_ebbupool_act_0_outliers,l1app_main_ebbupool_act_1_outliers,l1app_main_ebbupool_act_2_outliers,l1app_main_ebbupool_act_3_outliers,l1app_main_fh_main_poll-22_outliers,l1app_main_fh_rx_bbdev-21_outliers,phc2sys_phc2sys_outliers,ptp4l_ptp4l_outliers,in_octet_total_Ethernet15/2,in_octet_total_Ethernet31/1,out_octet_total_Ethernet15/2,out_octet_total_Ethernet31/1,anomaly'
        col_names = col_names.split(',')
        col_names 
        
        # col_names = ['timestamp','dumgr_du_1_DUMGR_LOGGER_1_max','dumgr_du_1_dumgr_du_1_max','gnb_cu_l3_l3_main_max','gnb_cu_pdcp_0_0_f1_worker_0_max','gnb_cu_pdcp_0_0_pdcp_master_0_max','gnb_cu_pdcp_0_0_pdcp_worker_0_max','gnb_cu_pdcp_0_0_recv_data_0_max','gnb_du_layer2_LowPrio_DU1_C0_max','gnb_du_layer2_TxCtrl_DU1_C0_max','gnb_du_layer2_f1_du_worker_0_max','gnb_du_layer2_pr_accumulator_max','gnb_du_layer2_rlcAccum_DU1_max','gnb_du_layer2_rlcTimer_DU1_max','gnb_du_layer2_rlcWorkrDU1__max','l1app_main_ebbupool_act_0_max','l1app_main_ebbupool_act_1_max','l1app_main_ebbupool_act_2_max','l1app_main_ebbupool_act_3_max','l1app_main_fh_main_poll-22_max','l1app_main_fh_rx_bbdev-21_max','phc2sys_phc2sys_max','ptp4l_ptp4l_max','dumgr_du_1_DUMGR_LOGGER_1_total_events','dumgr_du_1_dumgr_du_1_total_events','gnb_cu_l3_l3_main_total_events','gnb_cu_pdcp_0_0_f1_worker_0_total_events','gnb_cu_pdcp_0_0_pdcp_master_0_total_events','gnb_cu_pdcp_0_0_pdcp_worker_0_total_events','gnb_cu_pdcp_0_0_recv_data_0_total_events','gnb_du_layer2_LowPrio_DU1_C0_total_events','gnb_du_layer2_TxCtrl_DU1_C0_total_events','gnb_du_layer2_f1_du_worker_0_total_events','gnb_du_layer2_pr_accumulator_total_events','gnb_du_layer2_rlcAccum_DU1_total_events','gnb_du_layer2_rlcTimer_DU1_total_events','gnb_du_layer2_rlcWorkrDU1__total_events','l1app_main_ebbupool_act_0_total_events','l1app_main_ebbupool_act_1_total_events','l1app_main_ebbupool_act_2_total_events','l1app_main_ebbupool_act_3_total_events','l1app_main_fh_main_poll-22_total_events','l1app_main_fh_rx_bbdev-21_total_events','phc2sys_phc2sys_total_events','ptp4l_ptp4l_total_events','dumgr_du_1_DUMGR_LOGGER_1_total_runtime','dumgr_du_1_dumgr_du_1_total_runtime','gnb_cu_l3_l3_main_total_runtime','gnb_cu_pdcp_0_0_f1_worker_0_total_runtime','gnb_cu_pdcp_0_0_pdcp_master_0_total_runtime','gnb_cu_pdcp_0_0_pdcp_worker_0_total_runtime','gnb_cu_pdcp_0_0_recv_data_0_total_runtime','gnb_du_layer2_LowPrio_DU1_C0_total_runtime','gnb_du_layer2_TxCtrl_DU1_C0_total_runtime','gnb_du_layer2_f1_du_worker_0_total_runtime','gnb_du_layer2_pr_accumulator_total_runtime','gnb_du_layer2_rlcAccum_DU1_total_runtime','gnb_du_layer2_rlcTimer_DU1_total_runtime','gnb_du_layer2_rlcWorkrDU1__total_runtime','l1app_main_ebbupool_act_0_total_runtime','l1app_main_ebbupool_act_1_total_runtime','l1app_main_ebbupool_act_2_total_runtime','l1app_main_ebbupool_act_3_total_runtime','l1app_main_fh_main_poll-22_total_runtime','l1app_main_fh_rx_bbdev-21_total_runtime','phc2sys_phc2sys_total_runtime','ptp4l_ptp4l_total_runtime','dumgr_du_1_DUMGR_LOGGER_1_mean','dumgr_du_1_dumgr_du_1_mean','gnb_cu_l3_l3_main_mean','gnb_cu_pdcp_0_0_f1_worker_0_mean','gnb_cu_pdcp_0_0_pdcp_master_0_mean','gnb_cu_pdcp_0_0_pdcp_worker_0_mean','gnb_cu_pdcp_0_0_recv_data_0_mean','gnb_du_layer2_LowPrio_DU1_C0_mean','gnb_du_layer2_TxCtrl_DU1_C0_mean','gnb_du_layer2_f1_du_worker_0_mean','gnb_du_layer2_pr_accumulator_mean','gnb_du_layer2_rlcAccum_DU1_mean','gnb_du_layer2_rlcTimer_DU1_mean','gnb_du_layer2_rlcWorkrDU1__mean','l1app_main_ebbupool_act_0_mean','l1app_main_ebbupool_act_1_mean','l1app_main_ebbupool_act_2_mean','l1app_main_ebbupool_act_3_mean','l1app_main_fh_main_poll-22_mean','l1app_main_fh_rx_bbdev-21_mean','phc2sys_phc2sys_mean','ptp4l_ptp4l_mean','dumgr_du_1_DUMGR_LOGGER_1_range','dumgr_du_1_dumgr_du_1_range','gnb_cu_l3_l3_main_range','gnb_cu_pdcp_0_0_f1_worker_0_range','gnb_cu_pdcp_0_0_pdcp_master_0_range','gnb_cu_pdcp_0_0_pdcp_worker_0_range','gnb_cu_pdcp_0_0_recv_data_0_range','gnb_du_layer2_LowPrio_DU1_C0_range','gnb_du_layer2_TxCtrl_DU1_C0_range','gnb_du_layer2_f1_du_worker_0_range','gnb_du_layer2_pr_accumulator_range','gnb_du_layer2_rlcAccum_DU1_range','gnb_du_layer2_rlcTimer_DU1_range','gnb_du_layer2_rlcWorkrDU1__range','l1app_main_ebbupool_act_0_range','l1app_main_ebbupool_act_1_range','l1app_main_ebbupool_act_2_range','l1app_main_ebbupool_act_3_range','l1app_main_fh_main_poll-22_range','l1app_main_fh_rx_bbdev-21_range','phc2sys_phc2sys_range','ptp4l_ptp4l_range','dumgr_du_1_DUMGR_LOGGER_1_var','dumgr_du_1_dumgr_du_1_var','gnb_cu_l3_l3_main_var','gnb_cu_pdcp_0_0_f1_worker_0_var','gnb_cu_pdcp_0_0_pdcp_master_0_var','gnb_cu_pdcp_0_0_pdcp_worker_0_var','gnb_cu_pdcp_0_0_recv_data_0_var','gnb_du_layer2_LowPrio_DU1_C0_var','gnb_du_layer2_TxCtrl_DU1_C0_var','gnb_du_layer2_f1_du_worker_0_var','gnb_du_layer2_pr_accumulator_var','gnb_du_layer2_rlcAccum_DU1_var','gnb_du_layer2_rlcTimer_DU1_var','gnb_du_layer2_rlcWorkrDU1__var','l1app_main_ebbupool_act_0_var','l1app_main_ebbupool_act_1_var','l1app_main_ebbupool_act_2_var','l1app_main_ebbupool_act_3_var','l1app_main_fh_main_poll-22_var','l1app_main_fh_rx_bbdev-21_var','phc2sys_phc2sys_var','ptp4l_ptp4l_var','dumgr_du_1_DUMGR_LOGGER_1_std','dumgr_du_1_dumgr_du_1_std','gnb_cu_l3_l3_main_std','gnb_cu_pdcp_0_0_f1_worker_0_std','gnb_cu_pdcp_0_0_pdcp_master_0_std','gnb_cu_pdcp_0_0_pdcp_worker_0_std','gnb_cu_pdcp_0_0_recv_data_0_std','gnb_du_layer2_LowPrio_DU1_C0_std','gnb_du_layer2_TxCtrl_DU1_C0_std','gnb_du_layer2_f1_du_worker_0_std','gnb_du_layer2_pr_accumulator_std','gnb_du_layer2_rlcAccum_DU1_std','gnb_du_layer2_rlcTimer_DU1_std','gnb_du_layer2_rlcWorkrDU1__std','l1app_main_ebbupool_act_0_std','l1app_main_ebbupool_act_1_std','l1app_main_ebbupool_act_2_std','l1app_main_ebbupool_act_3_std','l1app_main_fh_main_poll-22_std','l1app_main_fh_rx_bbdev-21_std','phc2sys_phc2sys_std','ptp4l_ptp4l_std','dumgr_du_1_DUMGR_LOGGER_1_skewness','dumgr_du_1_dumgr_du_1_skewness','gnb_cu_l3_l3_main_skewness','gnb_cu_pdcp_0_0_f1_worker_0_skewness','gnb_cu_pdcp_0_0_pdcp_master_0_skewness','gnb_cu_pdcp_0_0_pdcp_worker_0_skewness','gnb_cu_pdcp_0_0_recv_data_0_skewness','gnb_du_layer2_LowPrio_DU1_C0_skewness','gnb_du_layer2_TxCtrl_DU1_C0_skewness','gnb_du_layer2_f1_du_worker_0_skewness','gnb_du_layer2_pr_accumulator_skewness','gnb_du_layer2_rlcAccum_DU1_skewness','gnb_du_layer2_rlcTimer_DU1_skewness','gnb_du_layer2_rlcWorkrDU1__skewness','l1app_main_ebbupool_act_0_skewness','l1app_main_ebbupool_act_1_skewness','l1app_main_ebbupool_act_2_skewness','l1app_main_ebbupool_act_3_skewness','l1app_main_fh_main_poll-22_skewness','l1app_main_fh_rx_bbdev-21_skewness','phc2sys_phc2sys_skewness','ptp4l_ptp4l_skewness','dumgr_du_1_DUMGR_LOGGER_1_kurtosis','dumgr_du_1_dumgr_du_1_kurtosis','gnb_cu_l3_l3_main_kurtosis','gnb_cu_pdcp_0_0_f1_worker_0_kurtosis','gnb_cu_pdcp_0_0_pdcp_master_0_kurtosis','gnb_cu_pdcp_0_0_pdcp_worker_0_kurtosis','gnb_cu_pdcp_0_0_recv_data_0_kurtosis','gnb_du_layer2_LowPrio_DU1_C0_kurtosis','gnb_du_layer2_TxCtrl_DU1_C0_kurtosis','gnb_du_layer2_f1_du_worker_0_kurtosis','gnb_du_layer2_pr_accumulator_kurtosis','gnb_du_layer2_rlcAccum_DU1_kurtosis','gnb_du_layer2_rlcTimer_DU1_kurtosis','gnb_du_layer2_rlcWorkrDU1__kurtosis','l1app_main_ebbupool_act_0_kurtosis','l1app_main_ebbupool_act_1_kurtosis','l1app_main_ebbupool_act_2_kurtosis','l1app_main_ebbupool_act_3_kurtosis','l1app_main_fh_main_poll-22_kurtosis','l1app_main_fh_rx_bbdev-21_kurtosis','phc2sys_phc2sys_kurtosis','ptp4l_ptp4l_kurtosis','dumgr_du_1_DUMGR_LOGGER_1_irq','dumgr_du_1_dumgr_du_1_irq','gnb_cu_l3_l3_main_irq','gnb_cu_pdcp_0_0_f1_worker_0_irq','gnb_cu_pdcp_0_0_pdcp_master_0_irq','gnb_cu_pdcp_0_0_pdcp_worker_0_irq','gnb_cu_pdcp_0_0_recv_data_0_irq','gnb_du_layer2_LowPrio_DU1_C0_irq','gnb_du_layer2_TxCtrl_DU1_C0_irq','gnb_du_layer2_f1_du_worker_0_irq','gnb_du_layer2_pr_accumulator_irq','gnb_du_layer2_rlcAccum_DU1_irq','gnb_du_layer2_rlcTimer_DU1_irq','gnb_du_layer2_rlcWorkrDU1__irq','l1app_main_ebbupool_act_0_irq','l1app_main_ebbupool_act_1_irq','l1app_main_ebbupool_act_2_irq','l1app_main_ebbupool_act_3_irq','l1app_main_fh_main_poll-22_irq','l1app_main_fh_rx_bbdev-21_irq','phc2sys_phc2sys_irq','ptp4l_ptp4l_irq','dumgr_du_1_DUMGR_LOGGER_1_outliers','dumgr_du_1_dumgr_du_1_outliers','gnb_cu_l3_l3_main_outliers','gnb_cu_pdcp_0_0_f1_worker_0_outliers','gnb_cu_pdcp_0_0_pdcp_master_0_outliers','gnb_cu_pdcp_0_0_pdcp_worker_0_outliers','gnb_cu_pdcp_0_0_recv_data_0_outliers','gnb_du_layer2_LowPrio_DU1_C0_outliers','gnb_du_layer2_TxCtrl_DU1_C0_outliers','gnb_du_layer2_f1_du_worker_0_outliers','gnb_du_layer2_pr_accumulator_outliers','gnb_du_layer2_rlcAccum_DU1_outliers','gnb_du_layer2_rlcTimer_DU1_outliers','gnb_du_layer2_rlcWorkrDU1__outliers','l1app_main_ebbupool_act_0_outliers','l1app_main_ebbupool_act_1_outliers','l1app_main_ebbupool_act_2_outliers','l1app_main_ebbupool_act_3_outliers','l1app_main_fh_main_poll-22_outliers','l1app_main_fh_rx_bbdev-21_outliers','phc2sys_phc2sys_outliers','ptp4l_ptp4l_outliers','ptp4l_ptp4l_others_runtime','phc2sys_phc2sys_others_runtime','l1app_main_fh_rx_bbdev-21_others_runtime','l1app_main_fh_main_poll-22_others_runtime','l1app_main_ebbupool_act_3_others_runtime','l1app_main_ebbupool_act_2_others_runtime','l1app_main_ebbupool_act_1_others_runtime','l1app_main_ebbupool_act_0_others_runtime','gnb_du_layer2_rlcWorkrDU1__others_runtime','gnb_du_layer2_rlcTimer_DU1_others_runtime','gnb_du_layer2_rlcAccum_DU1_others_runtime','gnb_du_layer2_pr_accumulator_others_runtime','gnb_du_layer2_f1_du_worker_0_others_runtime','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime','gnb_du_layer2_LowPrio_DU1_C0_others_runtime','gnb_cu_pdcp_0_0_recv_data_0_others_runtime','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime','gnb_cu_l3_l3_main_others_runtime','dumgr_du_1_dumgr_du_1_others_runtime','dumgr_du_1_DUMGR_LOGGER_1_others_runtime','ptp4l_ptp4l_max_runtime','phc2sys_phc2sys_max_runtime','l1app_main_fh_rx_bbdev-21_max_runtime','l1app_main_fh_main_poll-22_max_runtime','l1app_main_ebbupool_act_3_max_runtime','l1app_main_ebbupool_act_2_max_runtime','l1app_main_ebbupool_act_1_max_runtime','l1app_main_ebbupool_act_0_max_runtime','gnb_du_layer2_rlcWorkrDU1__max_runtime','gnb_du_layer2_rlcTimer_DU1_max_runtime','gnb_du_layer2_rlcAccum_DU1_max_runtime','gnb_du_layer2_pr_accumulator_max_runtime','gnb_du_layer2_f1_du_worker_0_max_runtime','gnb_du_layer2_TxCtrl_DU1_C0_max_runtime','gnb_du_layer2_LowPrio_DU1_C0_max_runtime','gnb_cu_pdcp_0_0_recv_data_0_max_runtime','gnb_cu_pdcp_0_0_pdcp_worker_0_max_runtime','gnb_cu_pdcp_0_0_pdcp_master_0_max_runtime','gnb_cu_pdcp_0_0_f1_worker_0_max_runtime','gnb_cu_l3_l3_main_max_runtime','dumgr_du_1_dumgr_du_1_max_runtime','dumgr_du_1_DUMGR_LOGGER_1_max_runtime','ptp4l_ptp4l_others_runtime_outliers','phc2sys_phc2sys_others_runtime_outliers','l1app_main_fh_rx_bbdev-21_others_runtime_outliers','l1app_main_fh_main_poll-22_others_runtime_outliers','l1app_main_ebbupool_act_3_others_runtime_outliers','l1app_main_ebbupool_act_2_others_runtime_outliers','l1app_main_ebbupool_act_1_others_runtime_outliers','l1app_main_ebbupool_act_0_others_runtime_outliers','gnb_du_layer2_rlcWorkrDU1__others_runtime_outliers','gnb_du_layer2_rlcTimer_DU1_others_runtime_outliers','gnb_du_layer2_rlcAccum_DU1_others_runtime_outliers','gnb_du_layer2_pr_accumulator_others_runtime_outliers','gnb_du_layer2_f1_du_worker_0_others_runtime_outliers','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_outliers','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_outliers','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_outliers','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_outliers','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_outliers','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_outliers','gnb_cu_l3_l3_main_others_runtime_outliers','dumgr_du_1_dumgr_du_1_others_runtime_outliers','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_outliers','ptp4l_ptp4l_others_runtime_irq','phc2sys_phc2sys_others_runtime_irq','l1app_main_fh_rx_bbdev-21_others_runtime_irq','l1app_main_fh_main_poll-22_others_runtime_irq','l1app_main_ebbupool_act_3_others_runtime_irq','l1app_main_ebbupool_act_2_others_runtime_irq','l1app_main_ebbupool_act_1_others_runtime_irq','l1app_main_ebbupool_act_0_others_runtime_irq','gnb_du_layer2_rlcWorkrDU1__others_runtime_irq','gnb_du_layer2_rlcTimer_DU1_others_runtime_irq','gnb_du_layer2_rlcAccum_DU1_others_runtime_irq','gnb_du_layer2_pr_accumulator_others_runtime_irq','gnb_du_layer2_f1_du_worker_0_others_runtime_irq','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_irq','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_irq','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_irq','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_irq','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_irq','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_irq','gnb_cu_l3_l3_main_others_runtime_irq','dumgr_du_1_dumgr_du_1_others_runtime_irq','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_irq','ptp4l_ptp4l_others_runtime_kurtosis','phc2sys_phc2sys_others_runtime_kurtosis','l1app_main_fh_rx_bbdev-21_others_runtime_kurtosis','l1app_main_fh_main_poll-22_others_runtime_kurtosis','l1app_main_ebbupool_act_3_others_runtime_kurtosis','l1app_main_ebbupool_act_2_others_runtime_kurtosis','l1app_main_ebbupool_act_1_others_runtime_kurtosis','l1app_main_ebbupool_act_0_others_runtime_kurtosis','gnb_du_layer2_rlcWorkrDU1__others_runtime_kurtosis','gnb_du_layer2_rlcTimer_DU1_others_runtime_kurtosis','gnb_du_layer2_rlcAccum_DU1_others_runtime_kurtosis','gnb_du_layer2_pr_accumulator_others_runtime_kurtosis','gnb_du_layer2_f1_du_worker_0_others_runtime_kurtosis','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_kurtosis','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_kurtosis','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_kurtosis','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_kurtosis','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_kurtosis','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_kurtosis','gnb_cu_l3_l3_main_others_runtime_kurtosis','dumgr_du_1_dumgr_du_1_others_runtime_kurtosis','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_kurtosis','ptp4l_ptp4l_others_runtime_skewness','phc2sys_phc2sys_others_runtime_skewness','l1app_main_fh_rx_bbdev-21_others_runtime_skewness','l1app_main_fh_main_poll-22_others_runtime_skewness','l1app_main_ebbupool_act_3_others_runtime_skewness','l1app_main_ebbupool_act_2_others_runtime_skewness','l1app_main_ebbupool_act_1_others_runtime_skewness','l1app_main_ebbupool_act_0_others_runtime_skewness','gnb_du_layer2_rlcWorkrDU1__others_runtime_skewness','gnb_du_layer2_rlcTimer_DU1_others_runtime_skewness','gnb_du_layer2_rlcAccum_DU1_others_runtime_skewness','gnb_du_layer2_pr_accumulator_others_runtime_skewness','gnb_du_layer2_f1_du_worker_0_others_runtime_skewness','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_skewness','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_skewness','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_skewness','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_skewness','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_skewness','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_skewness','gnb_cu_l3_l3_main_others_runtime_skewness','dumgr_du_1_dumgr_du_1_others_runtime_skewness','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_skewness','ptp4l_ptp4l_others_runtime_range','phc2sys_phc2sys_others_runtime_range','l1app_main_fh_rx_bbdev-21_others_runtime_range','l1app_main_fh_main_poll-22_others_runtime_range','l1app_main_ebbupool_act_3_others_runtime_range','l1app_main_ebbupool_act_2_others_runtime_range','l1app_main_ebbupool_act_1_others_runtime_range','l1app_main_ebbupool_act_0_others_runtime_range','gnb_du_layer2_rlcWorkrDU1__others_runtime_range','gnb_du_layer2_rlcTimer_DU1_others_runtime_range','gnb_du_layer2_rlcAccum_DU1_others_runtime_range','gnb_du_layer2_pr_accumulator_others_runtime_range','gnb_du_layer2_f1_du_worker_0_others_runtime_range','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_range','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_range','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_range','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_range','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_range','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_range','gnb_cu_l3_l3_main_others_runtime_range','dumgr_du_1_dumgr_du_1_others_runtime_range','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_range','ptp4l_ptp4l_others_runtime_std','phc2sys_phc2sys_others_runtime_std','l1app_main_fh_rx_bbdev-21_others_runtime_std','l1app_main_fh_main_poll-22_others_runtime_std','l1app_main_ebbupool_act_3_others_runtime_std','l1app_main_ebbupool_act_2_others_runtime_std','l1app_main_ebbupool_act_1_others_runtime_std','l1app_main_ebbupool_act_0_others_runtime_std','gnb_du_layer2_rlcWorkrDU1__others_runtime_std','gnb_du_layer2_rlcTimer_DU1_others_runtime_std','gnb_du_layer2_rlcAccum_DU1_others_runtime_std','gnb_du_layer2_pr_accumulator_others_runtime_std','gnb_du_layer2_f1_du_worker_0_others_runtime_std','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_std','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_std','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_std','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_std','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_std','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_std','gnb_cu_l3_l3_main_others_runtime_std','dumgr_du_1_dumgr_du_1_others_runtime_std','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_std','ptp4l_ptp4l_others_runtime_var','phc2sys_phc2sys_others_runtime_var','l1app_main_fh_rx_bbdev-21_others_runtime_var','l1app_main_fh_main_poll-22_others_runtime_var','l1app_main_ebbupool_act_3_others_runtime_var','l1app_main_ebbupool_act_2_others_runtime_var','l1app_main_ebbupool_act_1_others_runtime_var','l1app_main_ebbupool_act_0_others_runtime_var','gnb_du_layer2_rlcWorkrDU1__others_runtime_var','gnb_du_layer2_rlcTimer_DU1_others_runtime_var','gnb_du_layer2_rlcAccum_DU1_others_runtime_var','gnb_du_layer2_pr_accumulator_others_runtime_var','gnb_du_layer2_f1_du_worker_0_others_runtime_var','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_var','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_var','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_var','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_var','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_var','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_var','gnb_cu_l3_l3_main_others_runtime_var','dumgr_du_1_dumgr_du_1_others_runtime_var','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_var','ptp4l_ptp4l_others_runtime_mean','phc2sys_phc2sys_others_runtime_mean','l1app_main_fh_rx_bbdev-21_others_runtime_mean','l1app_main_fh_main_poll-22_others_runtime_mean','l1app_main_ebbupool_act_3_others_runtime_mean','l1app_main_ebbupool_act_2_others_runtime_mean','l1app_main_ebbupool_act_1_others_runtime_mean','l1app_main_ebbupool_act_0_others_runtime_mean','gnb_du_layer2_rlcWorkrDU1__others_runtime_mean','gnb_du_layer2_rlcTimer_DU1_others_runtime_mean','gnb_du_layer2_rlcAccum_DU1_others_runtime_mean','gnb_du_layer2_pr_accumulator_others_runtime_mean','gnb_du_layer2_f1_du_worker_0_others_runtime_mean','gnb_du_layer2_TxCtrl_DU1_C0_others_runtime_mean','gnb_du_layer2_LowPrio_DU1_C0_others_runtime_mean','gnb_cu_pdcp_0_0_recv_data_0_others_runtime_mean','gnb_cu_pdcp_0_0_pdcp_worker_0_others_runtime_mean','gnb_cu_pdcp_0_0_pdcp_master_0_others_runtime_mean','gnb_cu_pdcp_0_0_f1_worker_0_others_runtime_mean','gnb_cu_l3_l3_main_others_runtime_mean','dumgr_du_1_dumgr_du_1_others_runtime_mean','dumgr_du_1_DUMGR_LOGGER_1_others_runtime_mean' ,'in_octet_total_Ethernet15/2','in_octet_total_Ethernet31/1','out_octet_total_Ethernet15/2','out_octet_total_Ethernet31/1','anomaly']
        df_final = pd.DataFrame(columns=col_names)
        i = 0
        for col_name in col_names:
            if col_name in All_Platform_KPIs_and_Ethernet_Port_df.columns:
                df_final[col_name] = All_Platform_KPIs_and_Ethernet_Port_df[col_name]
            else:
                df_final[col_name] = 0
                i = i +1

        All_Platform_KPIs_and_Ethernet_Port_df = df_final
        return All_Platform_KPIs_and_Ethernet_Port_df


    def my_pivot(self,platform_df, col, new_colummn):
        data ={}
        for i in platform_df['Process_And_Thread'].unique():
            data[i] = []
        for i in platform_df['timestamp'].unique():
            tdf = platform_df[platform_df['timestamp'] == i]
            for j in tdf[col].unique():
                data[j].append({i:tdf[tdf[col]==j][new_colummn].values[0]})
            # Create a dictionary of DataFrames for each key
        dfs = {}
        for key, values in data.items():
            timestamp_data = {}
            for entry in values:
                timestamp, runtime = entry.popitem()
                timestamp_data[timestamp] = runtime
            df = pd.DataFrame.from_dict(timestamp_data, orient='index',columns=[f'{key}_{new_colummn}'])
            dfs[key] = df

        # Merge the DataFrames using outer join
        merged_df = pd.concat(list(dfs.values()), axis=1, join='outer')
        merged_df.fillna(0, inplace=True)
        # Display the resulting merged DataFrame
        return merged_df

    def merge_tables(self,source_table, tmp_df, KPI_Name):
        source_table = pd.merge_asof(left=source_table, right=tmp_df, on='timestamp',allow_exact_matches=True, direction="forward", tolerance=pd.Timedelta("200 milliseconds"))
        stats= {KPI_Name + ' Number before merging:':len(tmp_df),
                    KPI_Name + ' Number after merging:':len(source_table) -source_table[KPI_Name].isna().sum(),
                    KPI_Name + ' Succes merging rate:':round(source_table[KPI_Name].notna().sum()/len(tmp_df)*100,3)}

        return  source_table

    def parse_json(self, json_dict, json_object):
        
        # json_object = json.loads(json_object)    
        try:        
            if(json_object['stream_type']=='ebpf'):
                if(json_object['host_name']=='telco-2'):
                    try:
                        json_dict['ebpf'].append(self.parse_platform(json_object))
                    except:
                        json_dict['ebpf'] = []
                        json_dict['ebpf'].append(self.parse_platform(json_object))
            elif(json_object['stream_context_info']['hook_name']=='fapi_gnb_ul_config_req' ):
                try:
                    json_dict['fapi_gnb_ul_config_req'].append(self.parse_fapi_gnb_ul_config_req(json_object))
                except:
                    json_dict['fapi_gnb_ul_config_req'] = []
                    json_dict['fapi_gnb_ul_config_req'].append(self.parse_fapi_gnb_ul_config_req(json_object))
            elif(json_object['stream_context_info']['hook_name']=='mac_ul_crc_ind'):
                try:
                    json_dict['mac_ul_crc_ind'].append(self.parse_mac_ul_crc_ind(json_object))
                except:
                    json_dict['mac_ul_crc_ind'] = []
                    json_dict['mac_ul_crc_ind'].append(self.parse_mac_ul_crc_ind(json_object))
            # mac_sinr_update
            elif(json_object['stream_context_info']['hook_name']=='mac_sinr_update'):
                try:
                    json_dict['mac_sinr_update'].append(self.parse_mac_sinr_update(json_object))
                except:
                    json_dict['mac_sinr_update'] = []
                    json_dict['mac_sinr_update'].append(self.parse_mac_sinr_update(json_object))
            # mac_dl_harq
            elif(json_object['stream_context_info']['hook_name']=='mac_dl_harq'):
                try:
                    json_dict['mac_dl_harq'].append(self.parse_mac_dl_harq(json_object))
                except:
                    json_dict['mac_dl_harq'] = []
                    json_dict['mac_dl_harq'].append(self.parse_mac_dl_harq(json_object))
            # mac_csi_report
            elif(json_object['stream_context_info']['hook_name']=='mac_csi_report'):
                try:
                    json_dict['mac_csi_report'].append(self.parse_mac_csi_report(json_object))
                except:
                    json_dict['mac_csi_report'] = []
                    json_dict['mac_csi_report'].append(self.parse_mac_csi_report(json_object))
            # mac_bsr_update
            elif(json_object['stream_context_info']['hook_name']=='mac_bsr_update'):
                try:
                    json_dict['mac_bsr_update'].append(self.parse_mac_bsr_update(json_object))
                except:
                    json_dict['mac_bsr_update'] = []
                    json_dict['mac_bsr_update'].append(self.parse_mac_bsr_update(json_object))
            # fapi_gnb_dl_config_req
            elif(json_object['stream_context_info']['hook_name']=='fapi_gnb_dl_config_req'):
                try:
                    json_dict['fapi_gnb_dl_config_req'].append(self.parse_fapi_dl_config(json_object))
                except:
                    json_dict['fapi_gnb_dl_config_req'] = []
                    json_dict['fapi_gnb_dl_config_req'].append(self.parse_fapi_dl_config(json_object))
            # mhrx_ps
            elif(json_object['stream_context_info']['hook_name']=='midhaul_rx'):
                try:
                    json_dict['mhrx_ps'].append(self.parse_mhrx(json_object))
                except:
                    json_dict['mhrx_ps'] = []
                    json_dict['mhrx_ps'].append(self.parse_mhrx(json_object))
            # mhtx_ps
            elif(json_object['stream_context_info']['hook_name']=='midhaul_tx'):
                try:
                    json_dict['mhtx_ps'].append(self.parse_mhtx(json_object))
                except:
                    json_dict['mhtx_ps'] = []
                    json_dict['mhtx_ps'].append(self.parse_mhtx(json_object))
            # bhtx_ps
            elif(json_object['stream_context_info']['hook_name']=='backhaul_tx'):
                try:
                    json_dict['bhtx_ps'].append(self.parse_bhtx(json_object))
                except:
                    json_dict['bhtx_ps'] = []
                    json_dict['bhtx_ps'].append(self.parse_bhtx(json_object))
            # bhrx_ps
            elif(json_object['stream_context_info']['hook_name']=='backhaul_rx'):
                try:
                    json_dict['bhrx_ps'].append(self.parse_bhrx(json_object))
                except:
                    json_dict['bhrx_ps'] = []
                    json_dict['bhrx_ps'].append(self.parse_bhrx(json_object))    
            # rlc_mac_size
            elif(json_object['stream_context_info']['hook_name']=='rlc_mac_size'):
                try:
                    json_dict['rlc_mac_size'].append(self.parse_rlc_mac_size(json_object))
                except:
                    json_dict['rlc_mac_size'] = []
                    json_dict['rlc_mac_size'].append(self.parse_rlc_mac_size(json_object))
            # rlc_f1u_size
            elif(json_object['stream_context_info']['hook_name']=='rlc_f1u_size'):
                try:
                    json_dict['rlc_f1u_size'].append(self.parse_rlc_f1u_size(json_object))
                except:
                    json_dict['rlc_f1u_size'] = []
                    json_dict['rlc_f1u_size'].append(self.parse_rlc_f1u_size(json_object))
            # mac_rlc_size
            elif(json_object['stream_context_info']['hook_name']=='mac_rlc_size'):
                try:
                    json_dict['mac_rlc_size'].append(self.parse_mac_rlc_size(json_object))
                except:
                    json_dict['mac_rlc_size'] = []
                    json_dict['mac_rlc_size'].append(self.parse_mac_rlc_size(json_object))
            # mac_dl_bo_update
            elif(json_object['stream_context_info']['hook_name']=='mac_dl_bo_update'):
                try:
                    json_dict['mac_dl_bo_update'].append(self.parse_mac_dl_bo_update(json_object))
                except:
                    json_dict['mac_dl_bo_update'] = []
                    json_dict['mac_dl_bo_update'].append(self.parse_mac_dl_bo_update(json_object))
            # f1u_rlc_size
            elif(json_object['stream_context_info']['hook_name']=='f1u_rlc_size'):
                try:
                    json_dict['f1u_rlc_size'].append(self.parse_f1u_rlc_size(json_object))
                except:
                    json_dict['f1u_rlc_size'] = []
                    json_dict['f1u_rlc_size'].append(self.parse_f1u_rlc_size(json_object))
                     
            else:
                pass
        except KeyError:
            if(json_object['host_name']=='5gmgmt'):
                try:
                    json_dict['5gmgmt'].append(self.parse_swtich(json_object))
                except:
                    json_dict['5gmgmt'] = []
                    json_dict['5gmgmt'].append(self.parse_swtich(json_object))

        return json_dict

    def generate_csv(self, count, json_dict):
        source_table = []
        #parse platform
        ebpf = pd.DataFrame()
        fapi_gnb_ul_config_req = pd.DataFrame()
        mac_ul_crc_ind = pd.DataFrame()
        mac_sinr_update = pd.DataFrame()
        mac_dl_harq = pd.DataFrame()
        mac_csi_report = pd.DataFrame()
        mac_bsr_update = pd.DataFrame()
        fapi_gnb_dl_config_req = pd.DataFrame()
        mhrx_ps = pd.DataFrame()
        mhtx_ps = pd.DataFrame()
        bhtx_ps = pd.DataFrame()
        bhrx_ps = pd.DataFrame()
        rlc_mac_size = pd.DataFrame()
        rlc_f1u_size = pd.DataFrame()
        mac_rlc_size = pd.DataFrame()
        mac_dl_bo_update = pd.DataFrame()
        f1u_rlc_size = pd.DataFrame()
        mgmt = pd.DataFrame()
        All_Platform_KPIs_df = pd.DataFrame()

        #parse platform


        try:
            if(json_dict['ebpf']):
                # pprint.pprint(json_dict['ebpf'])
                json_dict['ebpf'] = [i for i in json_dict['ebpf'] if i != -1] 
                # print(len(json_dict['ebpf']))
                # pprint.pprint(json_dict['ebpf'])
                ebpf = pd.DataFrame(json_dict['ebpf'])
                ebpf['timestamp'] = pd.to_datetime(ebpf['timestamp'], format='%H:%M:%S.%f')
                ebpf=ebpf.sort_values(by='timestamp', ascending=True)
                ebpf.set_index('timestamp', inplace=True)
                ebpf = ebpf[~ebpf['thread_name'].str.startswith(('OS swapper', 'kworker', 'stress-ng', 'unknown', 'irq/', 'irq_work', 'ksoftirqd', 'rcuc/'))]
                ebpf['Process_And_Thread'] = ebpf['process_name'] + '_' + ebpf['thread_name']

                # If you want to reset the index after removing rows
                ebpf = ebpf.reset_index()

                # Sort by timestamp and Process_And_Thread column
                ebpf = ebpf.sort_values(by=['timestamp','Process_And_Thread'], ascending=True)

                # df.drop_duplicates(['timestamp', 'Process_And_Thread'], inplace=True)

                # calculate mean, var, std, ....outliers
                for index, row in ebpf.iterrows():
                    bin_value = row['bins']
                # print ('real max:',row['max'])
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.calc_platform_hist(bin_value)
                    ebpf.loc[index, 'mean']= mean
                    ebpf.loc[index, 'var']= var
                    ebpf.loc[index, 'std']= std
                    ebpf.loc[index, 'skewness']= skewness
                    ebpf.loc[index, 'kurtosis']= kurtosis
                    ebpf.loc[index, 'irq']= irq
                    ebpf.loc[index, 'outliers']= outliers
                    # =========== calc min and Range====================================
                    i = 0
                    bin_count = row['bins'][i]
                    range_start = 2 ** i
                    range_end = 2 ** (i + 1) - 1
                    range_size = range_end - range_start + 1
                    items_per_bin = bin_count / range_size
                    # Calculate total for this bin using formulas
                    Min_value = ((range_start + range_end) * range_size/2)*items_per_bin
                    # print ("range_start, range_end, bin_count, value:",range_start, range_end, bin_count, value)
                    ebpf.loc[index, 'range'] = row['max_runtime'] - Min_value
                    # print ("row['max'],  Min_value:",row['max'],  Min_value)
                # print(len(ebpf))
                # ============ calc min ===============================

                # histogram for sum for all threads
                for index, row in ebpf.iterrows():
                    bin_value = row['others_runtime_bin']
                    # print ('real max:',row['max'])
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.calc_platform_hist(bin_value)
                    ebpf.loc[index, 'others_runtime_mean']= mean
                    ebpf.loc[index, 'others_runtime_var']= var
                    ebpf.loc[index, 'others_runtime_std']= std
                    ebpf.loc[index, 'others_runtime_skewness']= skewness
                    ebpf.loc[index, 'others_runtime_kurtosis']= kurtosis
                    ebpf.loc[index, 'others_runtime_irq']= irq
                    ebpf.loc[index, 'others_runtime_outliers']= outliers
                    # =========== calc min and Range====================================
                    i = 0
                    bin_count = row['others_runtime_bin'][i]
                    range_start = 2 ** i
                    range_end = 2 ** (i + 1) - 1
                    range_size = range_end - range_start + 1
                    items_per_bin = bin_count / range_size
                    # Calculate total for this bin using formulas
                    Min_value = ((range_start + range_end) * range_size/2)*items_per_bin
                    # print ("range_start, range_end, bin_count, value:",range_start, range_end, bin_count, value)
                    ebpf.loc[index, 'others_runtime_range'] = row['max_runtime'] - Min_value
                # print(len(ebpf))
                df_list = []
                platform_mean_df = self.my_pivot(ebpf, 'Process_And_Thread', 'mean')
                # print(len(platform_mean_df))
                df_list.append(platform_mean_df)
                platform_var_df = self.my_pivot(ebpf, 'Process_And_Thread', 'var')
                # print(len(platform_var_df))
                df_list.append(platform_var_df)
                platform_std_df = self.my_pivot(ebpf, 'Process_And_Thread', 'std')
                # print(len(platform_std_df))
                df_list.append(platform_std_df)
                platform_range_df = self.my_pivot(ebpf, 'Process_And_Thread', 'range')
                # print(len(platform_range_df))
                df_list.append(platform_range_df)
                platform_skewness_df = self.my_pivot(ebpf, 'Process_And_Thread', 'skewness')
                # print(len(platform_skewness_df))
                df_list.append(platform_skewness_df)
                platform_kurtosis_df = self.my_pivot(ebpf, 'Process_And_Thread', 'kurtosis')
                # print(len(platform_kurtosis_df))
                df_list.append(platform_kurtosis_df)
                platform_irq_df = self.my_pivot(ebpf, 'Process_And_Thread', 'irq')
                # print(len(platform_irq_df))
                df_list.append(platform_irq_df)
                platform_outliers_df = self.my_pivot(ebpf, 'Process_And_Thread', 'outliers')
                # print(len(platform_outliers_df))
                df_list.append(platform_outliers_df)
                platform_others_runtime_mean_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_mean')
                # print(len(platform_others_runtime_mean_df))
                df_list.append(platform_others_runtime_mean_df)
                platform_others_runtime_var_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_var')
                # print(len(platform_others_runtime_var_df))
                df_list.append(platform_others_runtime_var_df)
            
                platform_others_runtime_std_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_std')
                # print(len(platform_others_runtime_std_df))
                df_list.append(platform_others_runtime_std_df)
                platform_others_runtime_range_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_range')
                # print(len(platform_others_runtime_range_df))
                df_list.append(platform_others_runtime_range_df)
                platform_others_runtime_skewness_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_skewness')
                # print(len(platform_others_runtime_skewness_df))
                df_list.append(platform_others_runtime_skewness_df)
                platform_others_runtime_kurtosis_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_kurtosis')
                # print(len(platform_others_runtime_kurtosis_df))
                df_list.append(platform_others_runtime_kurtosis_df)
                platform_others_runtime_irq_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_irq')
                # print(len(platform_others_runtime_irq_df))  
                df_list.append(platform_others_runtime_irq_df)
                platform_others_runtime_outliers_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_outliers')
                # print(len(platform_others_runtime_outliers_df))
                df_list.append(platform_others_runtime_outliers_df)

                platform_max_df = self.my_pivot(ebpf, 'Process_And_Thread', 'max')
                # print(len(platform_max_df))
                df_list.append(platform_max_df)
                platform_total_events_df = self.my_pivot(ebpf, 'Process_And_Thread', 'total_events')
                # print(len(platform_total_events_df))
                df_list.append(platform_total_events_df)
                platform_total_runtime_df = self.my_pivot(ebpf, 'Process_And_Thread', 'total_runtime')
                # print(len(platform_total_runtime_df))
                df_list.append(platform_total_runtime_df)
                platform_max_runtime_df = self.my_pivot(ebpf, 'Process_And_Thread', 'max_runtime')
                # print(len(platform_max_runtime_df))
                df_list.append(platform_max_runtime_df)
                platform_others_runtime_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime')
                # print(len(platform_others_runtime_df))
                df_list.append(platform_others_runtime_df)
                platform_anomaly_df = self.my_pivot(ebpf, 'Process_And_Thread', 'anomaly')
                # print(len(platform_anomaly_df))
                df_list.append(platform_anomaly_df)
                platform_anomaly_df.rename(columns={'gnb_cu_pdcp_0_0_pdcp_master_0_anomaly': 'anomaly'}, inplace=True)
                # print("all parsed") 
                # print(platform_others_runtime_df.columns)
                # print(len(df_list))

                # df_list = [platform_max_df, platform_total_events_df, platform_total_runtime_df,platform_mean_df, platform_range_df,platform_var_df,platform_std_df,platform_skewness_df ,platform_kurtosis_df,platform_irq_df,platform_outliers_df, platform_others_runtime_mean_df, platform_others_runtime_var_df, platform_others_runtime_std_df, platform_others_runtime_range_df, platform_others_runtime_skewness_df, platform_others_runtime_kurtosis_df, platform_others_runtime_irq_df, platform_others_runtime_outliers_df ,platform_max_runtime_df, platform_others_runtime_df, platform_anomaly_df['anomaly']]
                # print(len(df_list))
                All_Platform_KPIs_df = pd.concat(df_list , axis=1)
                # print(len(All_Platform_KPIs_df))
                All_Platform_KPIs_df.reset_index(inplace=True)
                All_Platform_KPIs_df.fillna(0, inplace=True)
                All_Platform_KPIs_df.rename(columns={'index': 'timestamp'}, inplace=True)
                # print(len(ebpf))
                # print(len(All_Platform_KPIs_df))  
        except KeyError:
            pass
            # pprint.pprint(json_dict['ebpf'])
            # exit(0)
        
        #parse fapi_gnb_ul_config_req
        try:
            if(json_dict['fapi_gnb_ul_config_req']):
                fapi_gnb_ul_config_req = pd.DataFrame(json_dict['fapi_gnb_ul_config_req'])
                fapi_gnb_ul_config_req['timestamp'] = fapi_gnb_ul_config_req['timestamp'].astype(np.int64)
                # print(len(source_table),len(fapi_gnb_ul_config_req))
                source_table.extend(fapi_gnb_ul_config_req['timestamp'].tolist())
                # print(len(source_table))
                fapi_gnb_ul_config_req['timestamp'] = pd.to_datetime(fapi_gnb_ul_config_req['timestamp'], unit='ns')
                fapi_gnb_ul_config_req=fapi_gnb_ul_config_req.sort_values(by='timestamp', ascending=True)
                fapi_gnb_ul_config_req.set_index('timestamp', inplace=True)
                # print(len(fapi_gnb_ul_config_req))
        except KeyError:
            pass
            # pprint.pprint(json_dict['fapi_gnb_ul_config_req'])
            # exit(0)
        
        #parse mac_ul_crc_ind
        try:
            if(json_dict['mac_ul_crc_ind']):
                mac_ul_crc_ind = pd.DataFrame(json_dict['mac_ul_crc_ind'])
                mac_ul_crc_ind['timestamp'] = mac_ul_crc_ind['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_ul_crc_ind))
                source_table.extend(mac_ul_crc_ind['timestamp'].tolist())
                # print(len(source_table))
                mac_ul_crc_ind['timestamp'] = pd.to_datetime(mac_ul_crc_ind['timestamp'], unit='ns')
                mac_ul_crc_ind=mac_ul_crc_ind.sort_values(by='timestamp', ascending=True)
                mac_ul_crc_ind.set_index('timestamp', inplace=True)
                # print(len(mac_ul_crc_ind))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_ul_crc_ind'])
            # exit(0)
        
        #parse mac_sinr_update
        try:
            if(json_dict['mac_sinr_update']):
                mac_sinr_update = pd.DataFrame(json_dict['mac_sinr_update'])
                mac_sinr_update['timestamp'] = mac_sinr_update['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_sinr_update))
                source_table.extend(mac_sinr_update['timestamp'].tolist())
                mac_sinr_update['timestamp'] = pd.to_datetime(mac_sinr_update['timestamp'], unit='ns')
                mac_sinr_update=mac_sinr_update.sort_values(by='timestamp', ascending=True)
                mac_sinr_update.set_index('timestamp', inplace=True)
                # print(len(mac_sinr_update))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_sinr_update'])
            # exit(0)
        
        #parse mac_dl_harq
        try:
            if(json_dict['mac_dl_harq']):
                mac_dl_harq = pd.DataFrame(json_dict['mac_dl_harq'])
                mac_dl_harq['timestamp'] = mac_dl_harq['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_dl_harq))
                source_table.extend(mac_dl_harq['timestamp'].tolist())
                mac_dl_harq['timestamp'] = pd.to_datetime(mac_dl_harq['timestamp'], unit='ns')
                mac_dl_harq=mac_dl_harq.sort_values(by='timestamp', ascending=True)
                mac_dl_harq.set_index('timestamp', inplace=True)
                # print(len(mac_dl_harq))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_dl_harq'])
            # exit(0)
        
        # parse mac_csi_report
        try:
            if(json_dict['mac_csi_report']):
                mac_csi_report = pd.DataFrame(json_dict['mac_csi_report'])
                mac_csi_report['timestamp'] = mac_csi_report['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_csi_report))
                source_table.extend(mac_csi_report['timestamp'].tolist())
                mac_csi_report['timestamp'] = pd.to_datetime(mac_csi_report['timestamp'], unit='ns')
                mac_csi_report=mac_csi_report.sort_values(by='timestamp', ascending=True)
                mac_csi_report.set_index('timestamp', inplace=True)
                # print(len(mac_csi_report))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_csi_report'])
            # exit(0)
        
        # parse mac_bsr_update
        try:
            if(json_dict['mac_bsr_update']):
                mac_bsr_update = pd.DataFrame(json_dict['mac_bsr_update'])
                mac_bsr_update['timestamp'] = mac_bsr_update['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_bsr_update))
                source_table.extend(mac_bsr_update['timestamp'].tolist())
                mac_bsr_update['timestamp'] = pd.to_datetime(mac_bsr_update['timestamp'], unit='ns')
                mac_bsr_update=mac_bsr_update.sort_values(by='timestamp', ascending=True)
                mac_bsr_update.set_index('timestamp', inplace=True)
                # print(len(mac_bsr_update))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_bsr_update'])
            # exit(0)
        
        # parse fapi_gnb_dl_config_req
        try:
            if(json_dict['fapi_gnb_dl_config_req']):
                fapi_gnb_dl_config_req = pd.DataFrame(json_dict['fapi_gnb_dl_config_req'])
                fapi_gnb_dl_config_req['timestamp'] = fapi_gnb_dl_config_req['timestamp'].astype(np.int64)
                # print(len(source_table),len(fapi_gnb_dl_config_req))
                source_table.extend(fapi_gnb_dl_config_req['timestamp'].tolist())
                fapi_gnb_dl_config_req['timestamp'] = pd.to_datetime(fapi_gnb_dl_config_req['timestamp'], unit='ns')
                fapi_gnb_dl_config_req=fapi_gnb_dl_config_req.sort_values(by='timestamp', ascending=True)
                fapi_gnb_dl_config_req.set_index('timestamp', inplace=True)
                # print(len(fapi_gnb_dl_config_req))
        except KeyError:
            pass
            # pprint.pprint(json_dict['fapi_gnb_dl_config_req'])
            # exit(0)
        # parse mhrx_ps
        try:
            if(json_dict['mhrx_ps']):

                mhrx_ps = pd.DataFrame(json_dict['mhrx_ps'])
                mhrx_ps['timestamp'] = mhrx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(mhrx_ps))
                source_table.extend(mhrx_ps['timestamp'].tolist())
                mhrx_ps['timestamp'] = pd.to_datetime(mhrx_ps['timestamp'], unit='ns')
                mhrx_ps=mhrx_ps.sort_values(by='timestamp', ascending=True)
                mhrx_ps.set_index('timestamp', inplace=True)
                # print(len(mhrx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mhrx_ps'])
            # exit(0)
        # parse mhtx_ps
        try:
            if(json_dict['mhtx_ps']):
                mhtx_ps = pd.DataFrame(json_dict['mhtx_ps'])
                mhtx_ps['timestamp'] = mhtx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(mhtx_ps))
                source_table.extend(mhtx_ps['timestamp'].tolist())
                mhtx_ps['timestamp'] = pd.to_datetime(mhtx_ps['timestamp'], unit='ns')
                mhtx_ps=mhtx_ps.sort_values(by='timestamp', ascending=True)
                mhtx_ps.set_index('timestamp', inplace=True)
                # print(len(mhtx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mhtx_ps'])
            # exit(0)
        # parse bhtx_ps
        try:
            if(json_dict['bhtx_ps']):
                bhtx_ps = pd.DataFrame(json_dict['bhtx_ps'])
                bhtx_ps['timestamp'] = bhtx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(bhtx_ps))
                source_table.extend(bhtx_ps['timestamp'].tolist())
                bhtx_ps['timestamp'] = pd.to_datetime(bhtx_ps['timestamp'], unit='ns')
                bhtx_ps=bhtx_ps.sort_values(by='timestamp', ascending=True)
                bhtx_ps.set_index('timestamp', inplace=True)
                # print(len(bhtx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['bhtx_ps'])
            # exit(0)
        
        # parse bhrx_ps
        try:
            if(json_dict['bhrx_ps']):
                bhrx_ps = pd.DataFrame(json_dict['bhrx_ps'])
                bhrx_ps['timestamp'] = bhrx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(bhrx_ps))
                source_table.extend(bhrx_ps['timestamp'].tolist())
                bhrx_ps['timestamp'] = pd.to_datetime(bhrx_ps['timestamp'], unit='ns')
                bhrx_ps=bhrx_ps.sort_values(by='timestamp', ascending=True)
                bhrx_ps.set_index('timestamp', inplace=True)
                # print(len(bhrx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['bhrx_ps'])
            # exit(0)

        # parse rlc_mac_size
        try:
            if(json_dict['rlc_mac_size']):
                rlc_mac_size = pd.DataFrame(json_dict['rlc_mac_size'])
                rlc_mac_size['timestamp'] = rlc_mac_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(rlc_mac_size))
                source_table.extend(rlc_mac_size['timestamp'].tolist())
                rlc_mac_size['timestamp'] = pd.to_datetime(rlc_mac_size['timestamp'], unit='ns')
                rlc_mac_size=rlc_mac_size.sort_values(by='timestamp', ascending=True)
                rlc_mac_size.set_index('timestamp', inplace=True)
                # print(len(rlc_mac_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['rlc_mac_size'])
            # exit(0)
        
        # parse rlc_f1u_size
        try:
            if(json_dict['rlc_f1u_size']):
                rlc_f1u_size = pd.DataFrame(json_dict['rlc_f1u_size'])
                rlc_f1u_size['timestamp'] = rlc_f1u_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(rlc_f1u_size))
                source_table.extend(rlc_f1u_size['timestamp'].tolist())
                rlc_f1u_size['timestamp'] = pd.to_datetime(rlc_f1u_size['timestamp'], unit='ns')
                rlc_f1u_size=rlc_f1u_size.sort_values(by='timestamp', ascending=True)
                rlc_f1u_size.set_index('timestamp', inplace=True)
                # print(len(rlc_f1u_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['rlc_f1u_size'])
            # exit(0)

        # parse mac_rlc_size
        try:
            if(json_dict['mac_rlc_size']):
                mac_rlc_size = pd.DataFrame(json_dict['mac_rlc_size'])
                mac_rlc_size['timestamp'] = mac_rlc_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_rlc_size))
                source_table.extend(mac_rlc_size['timestamp'].tolist())
                mac_rlc_size['timestamp'] = pd.to_datetime(mac_rlc_size['timestamp'], unit='ns')
                mac_rlc_size=mac_rlc_size.sort_values(by='timestamp', ascending=True)
                mac_rlc_size.set_index('timestamp', inplace=True)
                # print(len(mac_rlc_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_rlc_size'])
            # exit(0)
        # parse mac_dl_bo_update
        try:
            if(json_dict['mac_dl_bo_update']):
                mac_dl_bo_update = pd.DataFrame(json_dict['mac_dl_bo_update'])
                # print(len(source_table),len(mac_dl_bo_update))
                source_table.extend(mac_dl_bo_update['timestamp'].tolist())
                mac_dl_bo_update['timestamp'] = mac_dl_bo_update['timestamp'].astype(np.int64)
                mac_dl_bo_update['timestamp'] = pd.to_datetime(mac_dl_bo_update['timestamp'], unit='ns')
                mac_dl_bo_update=mac_dl_bo_update.sort_values(by='timestamp', ascending=True)
                mac_dl_bo_update.set_index('timestamp', inplace=True)
                # print(len(mac_dl_bo_update))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_dl_bo_update'])
            # exit(0)

        # parse f1u_rlc_size
        try:
            if(json_dict['f1u_rlc_size']):
                f1u_rlc_size = pd.DataFrame(json_dict['f1u_rlc_size'])
                f1u_rlc_size['timestamp'] = f1u_rlc_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(f1u_rlc_size))
                source_table.extend(f1u_rlc_size['timestamp'].tolist())
                f1u_rlc_size['timestamp'] = pd.to_datetime(f1u_rlc_size['timestamp'], unit='ns')
                f1u_rlc_size=f1u_rlc_size.sort_values(by='timestamp', ascending=True)
                f1u_rlc_size.set_index('timestamp', inplace=True)
                # print(len(f1u_rlc_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['f1u_rlc_size'])
            # exit(0)
        
        # parse 5gmgmt
        try:
            if(json_dict['5gmgmt']):
                json_dict['5gmgmt'] = [i for i in json_dict['5gmgmt'] if i != -1] 
                mgmt = pd.DataFrame(json_dict['5gmgmt'])
                mgmt['timestamp'] = pd.to_datetime(mgmt['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
                mgmt=mgmt.sort_values(by='timestamp', ascending=True)
                mgmt.set_index('timestamp', inplace=True)

                mgmt =pd.pivot_table(mgmt,
                    index='timestamp',
                    columns='name',
                    values=['in_octet_total', 'out_octet_total'])
                mgmt.columns = [f"{col[0]}_{col[1]}" for col in mgmt.columns]
                for i in mgmt.columns:
                    mgmt[i] = mgmt[i].diff(periods=2)
                    mgmt[i]*=(8/10) # divide by 10 [s] as we substract two time steps each one with 5 [s]
                    mgmt[i]/=1000*1000*1000
                mgmt = mgmt.reset_index(drop=False)
                
                
                # print(len(mgmt))
        except KeyError:
            pass
            # pprint.pprint(json_dict['5gmgmt'])
            # exit(0)
        # print(len(ebpf), len(fapi_gnb_ul_config_req), len(mac_ul_crc_ind), len(mac_sinr_update), len(mac_dl_harq), len(mac_csi_report), len(mac_bsr_update), len(fapi_gnb_dl_config_req), len(mhrx_ps), len(mhtx_ps), len(bhtx_ps), len(bhrx_ps), len(rlc_mac_size), len(rlc_f1u_size), len(mac_rlc_size), len(mac_dl_bo_update), len(f1u_rlc_size), len(mgmt))
        # print(len(source_table))
        source_table = list(set(source_table))
        
        source_table = pd.DataFrame(source_table, columns=['timestamp'])
        source_table['timestamp'] = source_table['timestamp'].astype(np.int64)
        source_table['timestamp'] = pd.to_datetime(source_table['timestamp'], unit='ns')
        source_table=source_table.sort_values(by='timestamp', ascending=True)
        # print(source_table)
        source_table = self.merge_tables(source_table, bhtx_ps[bhtx_ps['BHTX_In_size'].notna()].filter(regex='^BHTX_In_'),'BHTX_In_size')
        source_table = self.merge_tables(source_table, bhtx_ps[bhtx_ps['BHTX_Out_size'].notna()].filter(regex='^BHTX_Out_'),'BHTX_Out_size')
        source_table = self.merge_tables(source_table, bhrx_ps[bhrx_ps['BHRX_In_size'].notna()].filter(regex='^BHRX_In_'),'BHRX_In_size')
        source_table = self.merge_tables(source_table, bhrx_ps[bhrx_ps['BHRX_Out_size'].notna()].filter(regex='^BHRX_Out_'),'BHRX_Out_size')

        source_table = self.merge_tables(source_table, mhtx_ps[mhtx_ps['MHTX_In_size'].notna()].filter(regex='^MHTX_In_'),'MHTX_In_size')
        source_table = self.merge_tables(source_table, mhtx_ps[mhtx_ps['MHTX_Out_size'].notna()].filter(regex='^MHTX_Out_'),'MHTX_Out_size')
        source_table = self.merge_tables(source_table, mhrx_ps[mhrx_ps['MHRX_In_size'].notna()].filter(regex='^MHRX_In_'),'MHRX_In_size')
        source_table = self.merge_tables(source_table, mhrx_ps[mhrx_ps['MHRX_Out_size'].notna()].filter(regex='^MHRX_Out_'),'MHRX_Out_size')

        source_table = self.merge_tables(source_table, f1u_rlc_size[f1u_rlc_size['f1u_rlc_size'].notna()].filter(regex='^f1u_rlc_'),'f1u_rlc_size')

        source_table = self.merge_tables(source_table, rlc_f1u_size[rlc_f1u_size['rlc_f1u_size'].notna()].filter(regex='^rlc_f1u_'),'rlc_f1u_size')
        source_table = self.merge_tables(source_table, rlc_mac_size[rlc_mac_size['rlc_mac_size'].notna()].filter(regex='^rlc_mac_'),'rlc_mac_size')
        source_table = self.merge_tables(source_table, mac_rlc_size[mac_rlc_size['mac_rlc_size'].notna()].filter(regex='^mac_rlc_'),'mac_rlc_size')
        source_table = self.merge_tables(source_table, mac_bsr_update[mac_bsr_update['mac_bsr_update_max'].notna()].filter(regex='^mac_bsr_update_'),'mac_bsr_update_max')
        source_table = self.merge_tables(source_table, mac_csi_report[mac_csi_report['mac_csi_report_max'].notna()].filter(regex='^mac_csi_report_'),'mac_csi_report_max')
        source_table = self.merge_tables(source_table, mac_dl_harq[mac_dl_harq['mac_dl_harq_total'].notna()].filter(regex='^mac_dl_harq_'),'mac_dl_harq_total')
        source_table = self.merge_tables(source_table, mac_sinr_update[mac_sinr_update['mac_sinr_update_max'].notna()].filter(regex='^mac_sinr_update_'),'mac_sinr_update_max')
        source_table = self.merge_tables(source_table, mac_ul_crc_ind[mac_ul_crc_ind['mac_ul_CRC_Max'].notna()].filter(regex='^mac_ul_CRC_'),'mac_ul_CRC_Max')
        source_table = self.merge_tables(source_table, fapi_gnb_dl_config_req[fapi_gnb_dl_config_req['l1_tx'].notna()].filter(regex='^l1_'),'l1_tx')
        source_table = self.merge_tables(source_table, fapi_gnb_ul_config_req[fapi_gnb_ul_config_req['l1_UL_tx'].notna()].filter(regex='^l1_'),'l1_UL_tx')
        source_table = self.merge_tables(source_table, mac_dl_bo_update[mac_dl_bo_update['mac_dl_bo_max'].notna()].filter(regex='^mac_dl_bo_'),'mac_dl_bo_max')
        
        source_table.set_index('timestamp', inplace=True)
        source_table['anomaly'] = 0

        source_table.fillna(0, inplace=True)
        source_table.to_csv("radio"+str(count)+".csv")
        print(len(All_Platform_KPIs_df), len(mgmt))
        if(len(All_Platform_KPIs_df)>0 and len(mgmt)>0):
            platform = self.merge_platform(All_Platform_KPIs_df, mgmt)
            platform.to_csv("platform"+str(count)+".csv")             


    
    def generate_npy(self, count, json_dict):
        source_table = []
        #parse platform
        ebpf = pd.DataFrame()
        fapi_gnb_ul_config_req = pd.DataFrame()
        mac_ul_crc_ind = pd.DataFrame()
        mac_sinr_update = pd.DataFrame()
        mac_dl_harq = pd.DataFrame()
        mac_csi_report = pd.DataFrame()
        mac_bsr_update = pd.DataFrame()
        fapi_gnb_dl_config_req = pd.DataFrame()
        mhrx_ps = pd.DataFrame()
        mhtx_ps = pd.DataFrame()
        bhtx_ps = pd.DataFrame()
        bhrx_ps = pd.DataFrame()
        rlc_mac_size = pd.DataFrame()
        rlc_f1u_size = pd.DataFrame()
        mac_rlc_size = pd.DataFrame()
        mac_dl_bo_update = pd.DataFrame()
        f1u_rlc_size = pd.DataFrame()
        mgmt = pd.DataFrame()
        All_Platform_KPIs_df = pd.DataFrame()

        #parse platform


        try:
            if(json_dict['ebpf']):
                # pprint.pprint(json_dict['ebpf'])
                json_dict['ebpf'] = [i for i in json_dict['ebpf'] if i != -1] 
                # print(len(json_dict['ebpf']))
                # pprint.pprint(json_dict['ebpf'])
                ebpf = pd.DataFrame(json_dict['ebpf'])
                ebpf['timestamp'] = pd.to_datetime(ebpf['timestamp'], format='%H:%M:%S.%f')
                ebpf=ebpf.sort_values(by='timestamp', ascending=True)
                ebpf.set_index('timestamp', inplace=True)
                ebpf = ebpf[~ebpf['thread_name'].str.startswith(('OS swapper', 'kworker', 'stress-ng', 'unknown', 'irq/', 'irq_work', 'ksoftirqd', 'rcuc/'))]
                ebpf['Process_And_Thread'] = ebpf['process_name'] + '_' + ebpf['thread_name']

                # If you want to reset the index after removing rows
                ebpf = ebpf.reset_index()

                # Sort by timestamp and Process_And_Thread column
                ebpf = ebpf.sort_values(by=['timestamp','Process_And_Thread'], ascending=True)

                # df.drop_duplicates(['timestamp', 'Process_And_Thread'], inplace=True)

                # calculate mean, var, std, ....outliers
                for index, row in ebpf.iterrows():
                    bin_value = row['bins']
                # print ('real max:',row['max'])
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.calc_platform_hist(bin_value)
                    ebpf.loc[index, 'mean']= mean
                    ebpf.loc[index, 'var']= var
                    ebpf.loc[index, 'std']= std
                    ebpf.loc[index, 'skewness']= skewness
                    ebpf.loc[index, 'kurtosis']= kurtosis
                    ebpf.loc[index, 'irq']= irq
                    ebpf.loc[index, 'outliers']= outliers
                    # =========== calc min and Range====================================
                    i = 0
                    bin_count = row['bins'][i]
                    range_start = 2 ** i
                    range_end = 2 ** (i + 1) - 1
                    range_size = range_end - range_start + 1
                    items_per_bin = bin_count / range_size
                    # Calculate total for this bin using formulas
                    Min_value = ((range_start + range_end) * range_size/2)*items_per_bin
                    # print ("range_start, range_end, bin_count, value:",range_start, range_end, bin_count, value)
                    ebpf.loc[index, 'range'] = row['max_runtime'] - Min_value
                    # print ("row['max'],  Min_value:",row['max'],  Min_value)

                # ============ calc min ===============================

                # histogram for sum for all threads
                for index, row in ebpf.iterrows():
                    bin_value = row['others_runtime_bin']
                    # print ('real max:',row['max'])
                    mean, var ,std, skewness, kurtosis, irq,  outliers = self.calc_platform_hist(bin_value)
                    ebpf.loc[index, 'others_runtime_mean']= mean
                    ebpf.loc[index, 'others_runtime_var']= var
                    ebpf.loc[index, 'others_runtime_std']= std
                    ebpf.loc[index, 'others_runtime_skewness']= skewness
                    ebpf.loc[index, 'others_runtime_kurtosis']= kurtosis
                    ebpf.loc[index, 'others_runtime_irq']= irq
                    ebpf.loc[index, 'others_runtime_outliers']= outliers
                    # =========== calc min and Range====================================
                    i = 0
                    bin_count = row['others_runtime_bin'][i]
                    range_start = 2 ** i
                    range_end = 2 ** (i + 1) - 1
                    range_size = range_end - range_start + 1
                    items_per_bin = bin_count / range_size
                    # Calculate total for this bin using formulas
                    Min_value = ((range_start + range_end) * range_size/2)*items_per_bin
                    # print ("range_start, range_end, bin_count, value:",range_start, range_end, bin_count, value)
                    ebpf.loc[index, 'others_runtime_range'] = row['max_runtime'] - Min_value

                platform_mean_df = self.my_pivot(ebpf, 'Process_And_Thread', 'mean')
                platform_var_df = self.my_pivot(ebpf, 'Process_And_Thread', 'var')
                platform_std_df = self.my_pivot(ebpf, 'Process_And_Thread', 'std')
                platform_range_df = self.my_pivot(ebpf, 'Process_And_Thread', 'range')
                platform_skewness_df = self.my_pivot(ebpf, 'Process_And_Thread', 'skewness')
                platform_kurtosis_df = self.my_pivot(ebpf, 'Process_And_Thread', 'kurtosis')
                platform_irq_df = self.my_pivot(ebpf, 'Process_And_Thread', 'irq')
                platform_outliers_df = self.my_pivot(ebpf, 'Process_And_Thread', 'outliers')
                platform_others_runtime_mean_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_mean')
                platform_others_runtime_var_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_var')
                platform_others_runtime_std_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_std')
                platform_others_runtime_range_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_range')
                platform_others_runtime_skewness_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_skewness')
                platform_others_runtime_kurtosis_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_kurtosis')
                platform_others_runtime_irq_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_irq')
                platform_others_runtime_outliers_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime_outliers')
                platform_max_df = self.my_pivot(ebpf, 'Process_And_Thread', 'max')
                platform_total_events_df = self.my_pivot(ebpf, 'Process_And_Thread', 'total_events')
                platform_total_runtime_df = self.my_pivot(ebpf, 'Process_And_Thread', 'total_runtime')
                platform_max_runtime_df = self.my_pivot(ebpf, 'Process_And_Thread', 'max_runtime')
                platform_others_runtime_df = self.my_pivot(ebpf, 'Process_And_Thread', 'others_runtime')
                platform_anomaly_df = self.my_pivot(ebpf, 'Process_And_Thread', 'anomaly')
                platform_anomaly_df.rename(columns={'gnb_cu_pdcp_0_0_pdcp_master_0_anomaly': 'anomaly'}, inplace=True)
                All_Platform_KPIs_df = pd.concat([platform_max_df, platform_total_events_df, platform_total_runtime_df,platform_mean_df, platform_range_df,platform_var_df,platform_std_df,platform_skewness_df ,platform_kurtosis_df,platform_irq_df,platform_outliers_df, platform_others_runtime_mean_df, platform_others_runtime_var_df, platform_others_runtime_std_df, platform_others_runtime_range_df, platform_others_runtime_skewness_df, platform_others_runtime_kurtosis_df, platform_others_runtime_irq_df, platform_others_runtime_outliers_df ,platform_max_runtime_df, platform_others_runtime_df, platform_anomaly_df['anomaly']], axis=1)
                All_Platform_KPIs_df.reset_index(inplace=True)
                All_Platform_KPIs_df.fillna(0, inplace=True)
                All_Platform_KPIs_df.rename(columns={'index': 'timestamp'}, inplace=True)
                # print(len(ebpf))
        except KeyError:
            pass
            # pprint.pprint(json_dict['ebpf'])
            # exit(0)
        
        #parse fapi_gnb_ul_config_req
        try:
            if(json_dict['fapi_gnb_ul_config_req']):
                fapi_gnb_ul_config_req = pd.DataFrame(json_dict['fapi_gnb_ul_config_req'])
                fapi_gnb_ul_config_req['timestamp'] = fapi_gnb_ul_config_req['timestamp'].astype(np.int64)
                # print(len(source_table),len(fapi_gnb_ul_config_req))
                source_table.extend(fapi_gnb_ul_config_req['timestamp'].tolist())
                # print(len(source_table))
                fapi_gnb_ul_config_req['timestamp'] = pd.to_datetime(fapi_gnb_ul_config_req['timestamp'], unit='ns')
                fapi_gnb_ul_config_req=fapi_gnb_ul_config_req.sort_values(by='timestamp', ascending=True)
                fapi_gnb_ul_config_req.set_index('timestamp', inplace=True)
                # print(len(fapi_gnb_ul_config_req))
        except KeyError:
            pass
            # pprint.pprint(json_dict['fapi_gnb_ul_config_req'])
            # exit(0)
        
        #parse mac_ul_crc_ind
        try:
            if(json_dict['mac_ul_crc_ind']):
                mac_ul_crc_ind = pd.DataFrame(json_dict['mac_ul_crc_ind'])
                mac_ul_crc_ind['timestamp'] = mac_ul_crc_ind['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_ul_crc_ind))
                source_table.extend(mac_ul_crc_ind['timestamp'].tolist())
                # print(len(source_table))
                mac_ul_crc_ind['timestamp'] = pd.to_datetime(mac_ul_crc_ind['timestamp'], unit='ns')
                mac_ul_crc_ind=mac_ul_crc_ind.sort_values(by='timestamp', ascending=True)
                mac_ul_crc_ind.set_index('timestamp', inplace=True)
                # print(len(mac_ul_crc_ind))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_ul_crc_ind'])
            # exit(0)
        
        #parse mac_sinr_update
        try:
            if(json_dict['mac_sinr_update']):
                mac_sinr_update = pd.DataFrame(json_dict['mac_sinr_update'])
                mac_sinr_update['timestamp'] = mac_sinr_update['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_sinr_update))
                source_table.extend(mac_sinr_update['timestamp'].tolist())
                mac_sinr_update['timestamp'] = pd.to_datetime(mac_sinr_update['timestamp'], unit='ns')
                mac_sinr_update=mac_sinr_update.sort_values(by='timestamp', ascending=True)
                mac_sinr_update.set_index('timestamp', inplace=True)
                # print(len(mac_sinr_update))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_sinr_update'])
            # exit(0)
        
        #parse mac_dl_harq
        try:
            if(json_dict['mac_dl_harq']):
                mac_dl_harq = pd.DataFrame(json_dict['mac_dl_harq'])
                mac_dl_harq['timestamp'] = mac_dl_harq['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_dl_harq))
                source_table.extend(mac_dl_harq['timestamp'].tolist())
                mac_dl_harq['timestamp'] = pd.to_datetime(mac_dl_harq['timestamp'], unit='ns')
                mac_dl_harq=mac_dl_harq.sort_values(by='timestamp', ascending=True)
                mac_dl_harq.set_index('timestamp', inplace=True)
                # print(len(mac_dl_harq))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_dl_harq'])
            # exit(0)
        
        # parse mac_csi_report
        try:
            if(json_dict['mac_csi_report']):
                mac_csi_report = pd.DataFrame(json_dict['mac_csi_report'])
                mac_csi_report['timestamp'] = mac_csi_report['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_csi_report))
                source_table.extend(mac_csi_report['timestamp'].tolist())
                mac_csi_report['timestamp'] = pd.to_datetime(mac_csi_report['timestamp'], unit='ns')
                mac_csi_report=mac_csi_report.sort_values(by='timestamp', ascending=True)
                mac_csi_report.set_index('timestamp', inplace=True)
                # print(len(mac_csi_report))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_csi_report'])
            # exit(0)
        
        # parse mac_bsr_update
        try:
            if(json_dict['mac_bsr_update']):
                mac_bsr_update = pd.DataFrame(json_dict['mac_bsr_update'])
                mac_bsr_update['timestamp'] = mac_bsr_update['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_bsr_update))
                source_table.extend(mac_bsr_update['timestamp'].tolist())
                mac_bsr_update['timestamp'] = pd.to_datetime(mac_bsr_update['timestamp'], unit='ns')
                mac_bsr_update=mac_bsr_update.sort_values(by='timestamp', ascending=True)
                mac_bsr_update.set_index('timestamp', inplace=True)
                # print(len(mac_bsr_update))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_bsr_update'])
            # exit(0)
        
        # parse fapi_gnb_dl_config_req
        try:
            if(json_dict['fapi_gnb_dl_config_req']):
                fapi_gnb_dl_config_req = pd.DataFrame(json_dict['fapi_gnb_dl_config_req'])
                fapi_gnb_dl_config_req['timestamp'] = fapi_gnb_dl_config_req['timestamp'].astype(np.int64)
                # print(len(source_table),len(fapi_gnb_dl_config_req))
                source_table.extend(fapi_gnb_dl_config_req['timestamp'].tolist())
                fapi_gnb_dl_config_req['timestamp'] = pd.to_datetime(fapi_gnb_dl_config_req['timestamp'], unit='ns')
                fapi_gnb_dl_config_req=fapi_gnb_dl_config_req.sort_values(by='timestamp', ascending=True)
                fapi_gnb_dl_config_req.set_index('timestamp', inplace=True)
                # print(len(fapi_gnb_dl_config_req))
        except KeyError:
            pass
            # pprint.pprint(json_dict['fapi_gnb_dl_config_req'])
            # exit(0)
        # parse mhrx_ps
        try:
            if(json_dict['mhrx_ps']):

                mhrx_ps = pd.DataFrame(json_dict['mhrx_ps'])
                mhrx_ps['timestamp'] = mhrx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(mhrx_ps))
                source_table.extend(mhrx_ps['timestamp'].tolist())
                mhrx_ps['timestamp'] = pd.to_datetime(mhrx_ps['timestamp'], unit='ns')
                mhrx_ps=mhrx_ps.sort_values(by='timestamp', ascending=True)
                mhrx_ps.set_index('timestamp', inplace=True)
                # print(len(mhrx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mhrx_ps'])
            # exit(0)
        # parse mhtx_ps
        try:
            if(json_dict['mhtx_ps']):
                mhtx_ps = pd.DataFrame(json_dict['mhtx_ps'])
                mhtx_ps['timestamp'] = mhtx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(mhtx_ps))
                source_table.extend(mhtx_ps['timestamp'].tolist())
                mhtx_ps['timestamp'] = pd.to_datetime(mhtx_ps['timestamp'], unit='ns')
                mhtx_ps=mhtx_ps.sort_values(by='timestamp', ascending=True)
                mhtx_ps.set_index('timestamp', inplace=True)
                # print(len(mhtx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mhtx_ps'])
            # exit(0)
        # parse bhtx_ps
        try:
            if(json_dict['bhtx_ps']):
                bhtx_ps = pd.DataFrame(json_dict['bhtx_ps'])
                bhtx_ps['timestamp'] = bhtx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(bhtx_ps))
                source_table.extend(bhtx_ps['timestamp'].tolist())
                bhtx_ps['timestamp'] = pd.to_datetime(bhtx_ps['timestamp'], unit='ns')
                bhtx_ps=bhtx_ps.sort_values(by='timestamp', ascending=True)
                bhtx_ps.set_index('timestamp', inplace=True)
                # print(len(bhtx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['bhtx_ps'])
            # exit(0)
        
        # parse bhrx_ps
        try:
            if(json_dict['bhrx_ps']):
                bhrx_ps = pd.DataFrame(json_dict['bhrx_ps'])
                bhrx_ps['timestamp'] = bhrx_ps['timestamp'].astype(np.int64)
                # print(len(source_table),len(bhrx_ps))
                source_table.extend(bhrx_ps['timestamp'].tolist())
                bhrx_ps['timestamp'] = pd.to_datetime(bhrx_ps['timestamp'], unit='ns')
                bhrx_ps=bhrx_ps.sort_values(by='timestamp', ascending=True)
                bhrx_ps.set_index('timestamp', inplace=True)
                # print(len(bhrx_ps))
        except KeyError:
            pass
            # pprint.pprint(json_dict['bhrx_ps'])
            # exit(0)

        # parse rlc_mac_size
        try:
            if(json_dict['rlc_mac_size']):
                rlc_mac_size = pd.DataFrame(json_dict['rlc_mac_size'])
                rlc_mac_size['timestamp'] = rlc_mac_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(rlc_mac_size))
                source_table.extend(rlc_mac_size['timestamp'].tolist())
                rlc_mac_size['timestamp'] = pd.to_datetime(rlc_mac_size['timestamp'], unit='ns')
                rlc_mac_size=rlc_mac_size.sort_values(by='timestamp', ascending=True)
                rlc_mac_size.set_index('timestamp', inplace=True)
                # print(len(rlc_mac_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['rlc_mac_size'])
            # exit(0)
        
        # parse rlc_f1u_size
        try:
            if(json_dict['rlc_f1u_size']):
                rlc_f1u_size = pd.DataFrame(json_dict['rlc_f1u_size'])
                rlc_f1u_size['timestamp'] = rlc_f1u_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(rlc_f1u_size))
                source_table.extend(rlc_f1u_size['timestamp'].tolist())
                rlc_f1u_size['timestamp'] = pd.to_datetime(rlc_f1u_size['timestamp'], unit='ns')
                rlc_f1u_size=rlc_f1u_size.sort_values(by='timestamp', ascending=True)
                rlc_f1u_size.set_index('timestamp', inplace=True)
                # print(len(rlc_f1u_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['rlc_f1u_size'])
            # exit(0)

        # parse mac_rlc_size
        try:
            if(json_dict['mac_rlc_size']):
                mac_rlc_size = pd.DataFrame(json_dict['mac_rlc_size'])
                mac_rlc_size['timestamp'] = mac_rlc_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(mac_rlc_size))
                source_table.extend(mac_rlc_size['timestamp'].tolist())
                mac_rlc_size['timestamp'] = pd.to_datetime(mac_rlc_size['timestamp'], unit='ns')
                mac_rlc_size=mac_rlc_size.sort_values(by='timestamp', ascending=True)
                mac_rlc_size.set_index('timestamp', inplace=True)
                # print(len(mac_rlc_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_rlc_size'])
            # exit(0)
        # parse mac_dl_bo_update
        try:
            if(json_dict['mac_dl_bo_update']):
                mac_dl_bo_update = pd.DataFrame(json_dict['mac_dl_bo_update'])
                # print(len(source_table),len(mac_dl_bo_update))
                source_table.extend(mac_dl_bo_update['timestamp'].tolist())
                mac_dl_bo_update['timestamp'] = mac_dl_bo_update['timestamp'].astype(np.int64)
                mac_dl_bo_update['timestamp'] = pd.to_datetime(mac_dl_bo_update['timestamp'], unit='ns')
                mac_dl_bo_update=mac_dl_bo_update.sort_values(by='timestamp', ascending=True)
                mac_dl_bo_update.set_index('timestamp', inplace=True)
                # print(len(mac_dl_bo_update))
        except KeyError:
            pass
            # pprint.pprint(json_dict['mac_dl_bo_update'])
            # exit(0)

        # parse f1u_rlc_size
        try:
            if(json_dict['f1u_rlc_size']):
                f1u_rlc_size = pd.DataFrame(json_dict['f1u_rlc_size'])
                f1u_rlc_size['timestamp'] = f1u_rlc_size['timestamp'].astype(np.int64)
                # print(len(source_table),len(f1u_rlc_size))
                source_table.extend(f1u_rlc_size['timestamp'].tolist())
                f1u_rlc_size['timestamp'] = pd.to_datetime(f1u_rlc_size['timestamp'], unit='ns')
                f1u_rlc_size=f1u_rlc_size.sort_values(by='timestamp', ascending=True)
                f1u_rlc_size.set_index('timestamp', inplace=True)
                # print(len(f1u_rlc_size))
        except KeyError:
            pass
            # pprint.pprint(json_dict['f1u_rlc_size'])
            # exit(0)
        
        # parse 5gmgmt
        try:
            if(json_dict['5gmgmt']):
                json_dict['5gmgmt'] = [i for i in json_dict['5gmgmt'] if i != -1] 
                mgmt = pd.DataFrame(json_dict['5gmgmt'])
                mgmt['timestamp'] = pd.to_datetime(mgmt['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')
                mgmt=mgmt.sort_values(by='timestamp', ascending=True)
                mgmt.set_index('timestamp', inplace=True)

                mgmt =pd.pivot_table(mgmt,
                    index='timestamp',
                    columns='name',
                    values=['in_octet_total', 'out_octet_total'])
                mgmt.columns = [f"{col[0]}_{col[1]}" for col in mgmt.columns]
                for i in mgmt.columns:
                    mgmt[i] = mgmt[i].diff(periods=2)
                    mgmt[i]*=(8/10) # divide by 10 [s] as we substract two time steps each one with 5 [s]
                    mgmt[i]/=1000*1000*1000
                mgmt = mgmt.reset_index(drop=False)
                
                
                # print(len(mgmt))
        except KeyError:
            pass
            # pprint.pprint(json_dict['5gmgmt'])
            # exit(0)
        # print(len(ebpf), len(fapi_gnb_ul_config_req), len(mac_ul_crc_ind), len(mac_sinr_update), len(mac_dl_harq), len(mac_csi_report), len(mac_bsr_update), len(fapi_gnb_dl_config_req), len(mhrx_ps), len(mhtx_ps), len(bhtx_ps), len(bhrx_ps), len(rlc_mac_size), len(rlc_f1u_size), len(mac_rlc_size), len(mac_dl_bo_update), len(f1u_rlc_size), len(mgmt))
        # print(len(source_table))
        source_table = list(set(source_table))
        
        source_table = pd.DataFrame(source_table, columns=['timestamp'])
        source_table['timestamp'] = source_table['timestamp'].astype(np.int64)
        source_table['timestamp'] = pd.to_datetime(source_table['timestamp'], unit='ns')
        source_table=source_table.sort_values(by='timestamp', ascending=True)
        # print(source_table)
        source_table = self.merge_tables(source_table, bhtx_ps[bhtx_ps['BHTX_In_size'].notna()].filter(regex='^BHTX_In_'),'BHTX_In_size')
        source_table = self.merge_tables(source_table, bhtx_ps[bhtx_ps['BHTX_Out_size'].notna()].filter(regex='^BHTX_Out_'),'BHTX_Out_size')
        source_table = self.merge_tables(source_table, bhrx_ps[bhrx_ps['BHRX_In_size'].notna()].filter(regex='^BHRX_In_'),'BHRX_In_size')
        source_table = self.merge_tables(source_table, bhrx_ps[bhrx_ps['BHRX_Out_size'].notna()].filter(regex='^BHRX_Out_'),'BHRX_Out_size')

        source_table = self.merge_tables(source_table, mhtx_ps[mhtx_ps['MHTX_In_size'].notna()].filter(regex='^MHTX_In_'),'MHTX_In_size')
        source_table = self.merge_tables(source_table, mhtx_ps[mhtx_ps['MHTX_Out_size'].notna()].filter(regex='^MHTX_Out_'),'MHTX_Out_size')
        source_table = self.merge_tables(source_table, mhrx_ps[mhrx_ps['MHRX_In_size'].notna()].filter(regex='^MHRX_In_'),'MHRX_In_size')
        source_table = self.merge_tables(source_table, mhrx_ps[mhrx_ps['MHRX_Out_size'].notna()].filter(regex='^MHRX_Out_'),'MHRX_Out_size')

        source_table = self.merge_tables(source_table, f1u_rlc_size[f1u_rlc_size['f1u_rlc_size'].notna()].filter(regex='^f1u_rlc_'),'f1u_rlc_size')

        source_table = self.merge_tables(source_table, rlc_f1u_size[rlc_f1u_size['rlc_f1u_size'].notna()].filter(regex='^rlc_f1u_'),'rlc_f1u_size')
        source_table = self.merge_tables(source_table, rlc_mac_size[rlc_mac_size['rlc_mac_size'].notna()].filter(regex='^rlc_mac_'),'rlc_mac_size')
        source_table = self.merge_tables(source_table, mac_rlc_size[mac_rlc_size['mac_rlc_size'].notna()].filter(regex='^mac_rlc_'),'mac_rlc_size')
        source_table = self.merge_tables(source_table, mac_bsr_update[mac_bsr_update['mac_bsr_update_max'].notna()].filter(regex='^mac_bsr_update_'),'mac_bsr_update_max')
        source_table = self.merge_tables(source_table, mac_csi_report[mac_csi_report['mac_csi_report_max'].notna()].filter(regex='^mac_csi_report_'),'mac_csi_report_max')
        source_table = self.merge_tables(source_table, mac_dl_harq[mac_dl_harq['mac_dl_harq_total'].notna()].filter(regex='^mac_dl_harq_'),'mac_dl_harq_total')
        source_table = self.merge_tables(source_table, mac_sinr_update[mac_sinr_update['mac_sinr_update_max'].notna()].filter(regex='^mac_sinr_update_'),'mac_sinr_update_max')
        source_table = self.merge_tables(source_table, mac_ul_crc_ind[mac_ul_crc_ind['mac_ul_CRC_Max'].notna()].filter(regex='^mac_ul_CRC_'),'mac_ul_CRC_Max')
        source_table = self.merge_tables(source_table, fapi_gnb_dl_config_req[fapi_gnb_dl_config_req['l1_tx'].notna()].filter(regex='^l1_'),'l1_tx')
        source_table = self.merge_tables(source_table, fapi_gnb_ul_config_req[fapi_gnb_ul_config_req['l1_UL_tx'].notna()].filter(regex='^l1_'),'l1_UL_tx')
        source_table = self.merge_tables(source_table, mac_dl_bo_update[mac_dl_bo_update['mac_dl_bo_max'].notna()].filter(regex='^mac_dl_bo_'),'mac_dl_bo_max')
        
        source_table.set_index('timestamp', inplace=True)
        source_table['anomaly'] = 0

        source_table.fillna(0, inplace=True)
        source_table=source_table.iloc[:,1:-142]
        npdata = source_table.to_numpy()
        print(npdata.shape)
        qdata = npdata**0.2
        
        qdata = qdata[npdata.shape[0]%64:,:]
        qdata = qdata.reshape((-1,64,64))
        for i in range(qdata.shape[2]):
            qdata[:,:,i] = qdata[:,:,45]
        print(qdata.shape)
        print(qdata.max())
        np.save('radio'+str(count)+'.npy', qdata/22.8)

        
        # source_table.to_csv("radio"+str(count)+".csv")
        # if(len(All_Platform_KPIs_df)>0 and len(mgmt)>0):
        #     platform = self.merge_platform(All_Platform_KPIs_df, mgmt)
        #     platform.to_csv("platform"+str(count)+".csv")             
