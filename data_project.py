import pandas as pd
import numpy as np

class Data_Preprocess():
    def __init__(self,fname):
        self.df = pd.read_pickle(fname)

    def intro(self):
        print(self.df.columns)
        print(self.df.shape)
        print(self.df.dtypes)


    def describe_object(self,colname):
        print(self.df.loc[:,colname].value_counts())

    def describe_all(self,num):
        if np.issubdtype(self.df.iloc[:,num].dtype,object):
             self.describe_object(self.df.columns[num])
        else:
            print(self.df.iloc[:,num].describe())
        if num < len(self.df.columns)-1:
            return self.describe_all(num+1)


    def omit_zeros(self):
        self.df.drop(self.df[self.df.loc[:,'Value']==0].index,inplace=True)
        self.df.dropna(axis="index", how="any", inplace=True)
        self.df.reset_index(inplace=True,drop=True)
        return True


    def filt_top_areas_by_unit(self,u,n):
        self.df.drop(self.df[self.df.loc[:,"Unit"]!=u].index,inplace = True)
        ddf = self.df.loc[:,"Area"].value_counts(ascending=False).head(n)
        mask = self.df["Area"].isin([i for i in ddf.index])
        self.df = self.df[mask].reset_index(drop = True)


    def drop_cols(self,drop_lst):
        self.df.drop(columns = drop_lst,inplace= True)

    def calc_stats_by_factors(self,DataFrame,factors,vals,funcs):
        df_re = DataFrame.groupby(factors)
        return df_re[vals].agg(funcs)


    def norm_by_factors(self,cols):
        df_group = self.df.groupby(cols)
        add_to = df_group["Value"].transform(lambda x:((x-x.mean())/x.std()))
        self.df = pd.concat([self.df,add_to.rename("normed_val")],axis=1)


    def split_by_factor(self,factor,val):
        temp_df = self.df[self.df.loc[:,factor] == val].reset_index(drop = True)
        return temp_df

    def merge_dfs(self,s1,s2,colnames):
        return pd.concat([s1,s2],keys=colnames,join="inner",axis=1)

    def diff_cols(self,dat,col1,col2):
        return dat[col1]-(dat[col2])

    def apply_diff_cols(self,d,c1,c2,newcol):
        n_d = d.apply(func=self.diff_cols,axis=1,args=(c1,c2))
        return pd.concat([d,n_d.rename(newcol)],axis=1)

def main():
    data = Data_Preprocess('data.pickle')
    # 1.
    data.intro()

    # 2. + 3.
    data.describe_all(0)

    # 4.
    data.omit_zeros()
    print("4. after omitting zeros df shape is ", data.df.shape)
    print(data.df.head())

    # 5.
    data.filt_top_areas_by_unit('tonnes',5)
    print("5. After filtering product tonnes from 5 most reported areas, df shape is ", data.df.shape)
    print(data.df.head())

    # 6.
    data.drop_cols(["Item","Unit"])
    print("df shape after cols reduction",data.df.shape)

    # 7.
    # group by year+element, calculate annual mean and std of export and import quantities
    print("annual mean and std of export and import quantities")
    print(data.calc_stats_by_factors(data.df,["Year","Element"],"Value",[np.mean,np.std]))

    # 8.
    # apply z-score normalization by Year
    data.norm_by_factors(["Year"])
    print("after z-score normalization by Year\n")
    print(data.df.head(10))
    print(data.df.shape)

    # 9.
    data.export_df = data.split_by_factor("Element","Export Quantity")
    print("Export dataframe shape is ",data.export_df.shape)
    print("Export dataframe head:\n", data.export_df.head())
    data.import_df = data.split_by_factor("Element", "Import Quantity")
    print("Import dataframe shape is ",data.import_df.shape)
    print("Import dataframe head:\n", data.import_df.head())

    # Additional tests on export/import data
    data.export_df = data.calc_stats_by_factors(data.export_df,["Area","Year"],"Value",np.mean)
    print("export dataframe stats shape is ",data.export_df.shape)
    print(data.export_df.head())
    data.import_df = data.calc_stats_by_factors(data.import_df,["Area","Year"],"Value",np.mean)
    print("import dataframe stats shape is ",data.import_df.shape)
    print(data.import_df.head())

    # 10.
    data.merged = data.merge_dfs(data.import_df,data.export_df,["Import","Export"])
    print("merged Export-Import dataframe shape is ",data.merged.shape)
    print("merged Export-Import head:\n",data.merged.head())

    # 11.
    data.merged = data.apply_diff_cols(data.merged,c1="Export",c2="Import",newcol='GNI')
    print("merged Export-Import dataframe with GNI looks like this:")
    print(data.merged.head(10))


if __name__=="__main__":
    main()