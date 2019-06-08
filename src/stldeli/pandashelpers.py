# coding: utf-8

# In[1]:


import re
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from fractions import Fraction


# In[2]:


def clean_column_names(self):
    new_column_names = {old: re.sub(string=old.lower(),
                                    pattern=r'\W',  # \W matches non-alphnumeric
                                    repl='_').strip('_')
                        for old in self.columns
                        }
    return self.rename(columns=new_column_names)


pd.DataFrame.clean_column_names = clean_column_names


# In[3]:


def parse_date_columns(self):
    for date_column in self.filter(regex="date").columns:
        self[date_column] = pd.to_datetime(self[date_column])
    return self


pd.DataFrame.parse_date_columns = parse_date_columns


# In[4]:


def zero_to_null(self, subset):
    for column in subset:
        self[column] = self[column].apply(lambda x: x if x != 0 else np.nan)
    return self


pd.DataFrame.zero_to_null = zero_to_null


# In[5]:


def merge_multi(self, df, **kwargs):
    try:
        left = self.reset_index()
    except ValueError:
        left = self.reset_index(drop=True)

    try:
        right = df.reset_index()
    except ValueError:
        right = df.reset_index(drop=True)

    return left.merge(right,
                      **kwargs) \
        .set_index(self.index.names)


pd.DataFrame.merge_multi = merge_multi


# In[6]:


def deduplicate(self, key, numeric_agg='max', non_numeric_agg='first', override=dict()):
    how_to_agg = {index: numeric_agg if np.issubdtype(value, np.number) else non_numeric_agg
                  for (index, value) in self.dtypes.iteritems()
                  }
    how_to_agg.update(override)
    return self.groupby(key).agg(how_to_agg)


pd.DataFrame.deduplicate = deduplicate


# In[7]:


def parse_api_columns(self):
    for api_column in self.filter(regex="api").columns:
        self[api_column] = self[api_column].apply(str).str.replace(r'\W', '').str.pad(14
                                                                                      , side='right'
                                                                                      , fillchar='0'
                                                                                      )
    return self


pd.DataFrame.parse_api_columns = parse_api_columns

if __name__ == '__main__':
    # In[8]:

    casing = pd.concat(map(pd.read_csv
                           , glob.glob('./data/welldatabase/casing/*.csv')
                           )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[9]:

    completion = pd.concat(map(pd.read_csv
                               , glob.glob('./data/welldatabase/completion/*.csv')
                               )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[10]:

    directional = pd.concat(map(pd.read_csv
                                , glob.glob('./data/welldatabase/directional/*.csv')
                                )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[11]:

    formation = pd.concat(map(pd.read_csv
                              , glob.glob('./data/welldatabase/formation/*.csv')
                              )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[12]:

    fracstage = pd.concat(map(pd.read_csv
                              , glob.glob('./data/welldatabase/fracstage/*.csv')
                              )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[13]:

    header = pd.concat(map(pd.read_csv
                           , glob.glob('./data/welldatabase/header/*.csv')
                           )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[14]:

    perf = pd.concat(map(pd.read_csv
                         , glob.glob('./data/welldatabase/perf/*.csv')
                         )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[15]:

    production = pd.concat(map(pd.read_csv
                               , glob.glob('./data/welldatabase/production/*.csv')
                               )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[16]:

    productionsummary = pd.concat(map(pd.read_csv
                                      , glob.glob('./data/welldatabase/productionsummary/*.csv')
                                      )) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()

    # In[17]:

    test = pd.concat(map(pd.read_csv
                         , glob.glob('./data/welldatabase/test/*.csv')
                         )
                     ) \
        .clean_column_names() \
        .parse_date_columns() \
        .parse_api_columns()


    # In[18]:

    def string_to_fraction(x):
        try:
            result = float(Fraction(x))
        except AttributeError:
            result = float(x)
        except TypeError:
            result = np.nan
        except ValueError:
            result = np.nan
        while result > 1:
            result = result / 64
        return result


    # In[19]:

    test['chokesize_clean'] = test['chokesize'].str.replace('-', '/').str.replace('TH', '').str.strip('"').str.strip(
        "'").str.replace(pat="OPEN|NONE|FO|OPEN FLOW", repl='1').str.replace(pat="0|CLOSED|INSERT", repl='').apply(
        string_to_fraction)

    # In[20]:

    tubingandpacker = pd.concat(map(pd.read_csv
                                    , glob.glob('./data/welldatabase/tubingandcasing/*.csv')
                                    )) \
        .clean_column_names() \
        .parse_date_columns()

    # In[21]:

    header = header[header['wellboreprofile'] == 'HORIZONTAL']
    sqdist = (header['surfacelatitude'] - header['bottomholelatitude']) ** 2 + (
            header['surfacelongitude'] - header['bottomholelongitude']) ** 2
    header['surface_to_bottomhole_distance'] = sqdist.map(lambda x: np.sqrt(x))

    # In[22]:

    linear_reg = LinearRegression()
    xy = header.loc[
        header['surface_to_bottomhole_distance'] != 0, ['laterallength', 'surface_to_bottomhole_distance']].dropna()
    y = xy['laterallength'].values.reshape(-1, 1)
    X = xy['surface_to_bottomhole_distance'].values.reshape(-1, 1)
    linear_reg.fit(X=X, y=y)

    # In[23]:

    plt.plot(X, y, 'o'
             , X, linear_reg.predict(X), '-k')

    # In[24]:

    header['laterallength_from_bottom'] = linear_reg.predict(
        header['surface_to_bottomhole_distance'].values.reshape(-1, 1))

    # In[25]:

    header['missing_laterallength'] = header['laterallength'].isnull()

    # In[26]:

    header_summary = header.deduplicate('api', override={'missing_laterallength': 'all'})

    # In[27]:

    completion = zero_to_null(completion, ['upperperf', 'lowerperf'])

    # In[28]:

    completion_summary = completion.deduplicate(['api', 'completiondate'])

    # In[29]:

    index_of_deepest_top = formation.groupby("api").agg({'topdepth': 'idxmax'})['topdepth'].dropna()
    formation_summary = formation.iloc[index_of_deepest_top].groupby('api').agg({'name': 'first'})

    # In[30]:

    fracstage_summary = fracstage.deduplicate('api')

    # In[31]:

    perf = zero_to_null(perf, ['lowerperf', 'upperperf'])

    # In[32]:

    perf_summary = perf.deduplicate('api')

    # In[33]:

    production['yearmonth'] = production['date'].apply(
        lambda x: '{YEAR}-{MONTH:02d}'.format(YEAR=x.year, MONTH=x.month))

    # In[34]:

    production = production[production['yearmonth'] > '2011-01']

    # In[35]:

    production['days'] = pd.to_timedelta(production['days'], unit='D')
    production['first_producing_day_of_month'] = production['date'] + pd.DateOffset(months=1) - production['days']

    # In[36]:

    production_summary = production.deduplicate(['api', 'yearmonth'])

    # In[37]:

    test_summary = test.deduplicate(['api', 'testdate'])

    # In[38]:

    productionsummary_summary = productionsummary.deduplicate('api')

    # In[39]:

    combined = header_summary.join(productionsummary_summary, rsuffix='prodsum').join(perf_summary,
                                                                                      rsuffix='perf').join(
        fracstage_summary, rsuffix='frac').merge_multi(test_summary, suffixes=('', 'test')).merge_multi(
        completion_summary, suffixes=('', 'comp')).merge_multi(production_summary, suffixes=('', 'prod'))

    # In[40]:

    combined['days_since_completion'] = (combined['first_producing_day_of_month'] - combined[
        'completiondate']) / np.timedelta64(1, 'D')

    # In[41]:

    combined['increment_30days'] = combined['days_since_completion'].apply(lambda x: np.floor(x / 30) + 1)

    # In[42]:

    combined = combined[combined['first_producing_day_of_month'] >= combined['completiondate']]
    combined.shape

    # In[43]:

    combined_subset = combined[combined['increment_30days'].map(lambda x: 0 <= x <= 12)]
    combined_subset.shape

    # In[44]:

    combined_subset = combined_subset.deduplicate(key=['api', 'increment_30days']).deduplicate(key=['api'],
                                                                                               override={'oil': 'sum'})

    combined_subset.shape

    # In[45]:

    combined_subset.to_csv('./data/welldb_combined.csv')
