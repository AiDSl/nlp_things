#this is pure source code,if you want can use it.
import jieba;
jieba.enable_paddle();

import copy;
import math;

import matplotlib.pyplot as plt;

def homework1():
    print("start scaning dataset!");
    result=tridf("./hw1-dataset.txt");
    #result=tridf("./test.txt");
    print("done!");
    
    #fisrt data
    print("preparing first dataset");
    fig1={"x":[],"y":[]};
    hihg_frquency_data=sort_and_gen_dataset(result["stat"].items(),lambda i:i[1]["count"],100);
    for entry in hihg_frquency_data:
        fig1["x"].append(clear_text(entry[0]));
        fig1["y"].append(entry[1]["count"]/result["total_word_count"]);
    print("done!")
    
    #second data
    print("preparing second dataset");
    fig2={"x":[],"y":[]};
    full_stat=[];
    for line,doc in result["doc"].items():
        full_stat.extend(doc["stat"].values());
    high_weight_data=sort_and_gen_dataset(full_stat,lambda i:i["tfidf"],100);
    for entry in high_weight_data:
        fig2["x"].append(clear_text(entry["source"])+"\n的字詞:"+clear_text(entry["name"]));
        fig2["y"].append(entry["tfidf"]);
    print("done!")
    
    #printing result
    print("readying for the drawing");
    plot_setting();
    print("let`s start drawing!");
    draw_plot("frquency of words",fig1["x"],fig1["y"],hihg_frquency_data);
    draw_plot("tf-idf of word from all document",fig2["x"],fig2["y"],high_weight_data);
    print("everything have been done!")
    return;
#end of homework1



def plot_setting():
    plt.rcParams["figure.figsize"] = (300,20);
    plt.rcParams['font.sans-serif']="Microsoft JhengHei";#微軟正黑體
    #plt.rcParams['font.sans-serif']="DFKai-SB";#標楷體
    plt.rcParams.update({'font.size': 40})
    return;
#end of plot_setting

def draw_plot(inname,inx,iny,inraw_data,):
    print("=====================================");
    print(inname);
    plt.suptitle(inname);
    plt.xticks(rotation=80,horizontalalignment='left')
    plt.yticks(rotation=0)
    plt.plot(inx,iny)
    plt.show();
    print("the raw data:");
    print(inraw_data);
    print("\n\n\n\n\n\n\n");
    return;
#end of draw_plot
    

def sort_and_gen_dataset(rawdata,inlambda,insize):
    stat_view=sorted(rawdata,reverse=True,key=inlambda);
    return stat_view[0:insize];
#end of sort_and_gen_dataset

def tridf(infilename):
    doc_list={};
    with open(infilename,"rt",encoding='utf-8') as ifile:
        for line in ifile :
            list_of_line=jieba.cut(line);
            doc_list[line]=list_of_line; 
    result=scan_doc_list(doc_list);
    return result;
#tridf ends

def scan_doc_list(inlist):
    docs={
        "total_doc_count":0,
        "total_word_count":0,
        "stat":{},# word -> {count,idf}
        "doc":{}#line -> doc
    };
    
    #scan every doc in lista
    for line,line_of_list in inlist.items():
        doc=scan_doc(line,line_of_list);
        docs["doc"][line]=doc;
        #merge stat loop
        for word,info in doc["stat"].items():
            if word in docs["stat"]:
                docs["stat"][word]["count"]+=doc["stat"][word]["count"];
                docs["stat"][word]["doc_count"]+=1;
            else:
                docs["stat"][word]={};
                docs["stat"][word]["name"]=word;
                docs["stat"][word]["count"]=copy.deepcopy(doc["stat"][word]["count"]);
                docs["stat"][word]["doc_count"]=1;
        #end of merge loop
        docs["total_doc_count"]+=1;
        docs["total_word_count"]+=doc["total_word_count"];
    #end of scan
    
    for word,info in docs["stat"].items():
        info["idf"]=math.log((docs["total_doc_count"])/(info["doc_count"]),10);
    #end of idf cal
    
    for word,doc in docs["doc"].items():
        for word,info in doc["stat"].items():
            tr=(info["count"])/(doc["total_word_count"]);
            idf=docs["stat"][word]["idf"];
            info["tfidf"]=(tr)*(idf);
        #end of a word
    #end of a doc
    return docs;
#end of scan_doc_list

def scan_doc(inname,inlist):
    doc={
        "name":inname,
        "total_word_count":0,
        "stat":{}# word -> {count,idf}
    };
    
    #scan thorogh list
    for word in inlist:
        if word in doc["stat"]:
            doc["stat"][word]["count"]+=1;
        else:
            doc["stat"][word]={};
            doc["stat"][word]["name"]=word;
            doc["stat"][word]["source"]=inname;
            doc["stat"][word]["count"]=1;
        doc["total_word_count"]+=1;
    #end of loop
   
    return doc;
#end of scan_doc


def clear_text(intext):
    first_escape=' '.join(intext.split());
    second_escape='\\$'.join(first_escape.split('$'));
    return second_escape;
#end of clear_text






homework1();