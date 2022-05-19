import altair as alt
import pandas as pd
import streamlit as st
from vega_datasets import data
import pandas_datareader as pdr
from datetime import datetime
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

from pathlib import Path
import base64
import time
from datetime import date, datetime
from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
from fbprophet import Prophet
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error





st.set_page_config(
    page_title="Time Series Playground by EDHEC ", layout="wide", page_icon="./images/flask.png"
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

images = Image.open('images/edhec.png')
st.image(images, width=200)

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

st.sidebar.header("Select Stock")
list_symbols = ['AAPL', 'AMZN','GOOG', 'IBM','MSFT','FB','TSLA','NVDA',
                    'PG','JPM','WMT','CVX','BAC','PFE']
symbols = st.sidebar.multiselect("", list_symbols, list_symbols[:5])


st.sidebar.header("Select KPI")
list_kpi = ['High', 'Low','Open','Close','Volume','Adj Close']
kpi = st.sidebar.selectbox("", list_kpi)





@st.experimental_memo
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source


@st.experimental_memo(ttl=60 * 60 * 24)
def get_chart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="date",
            y=kpi,
            color="symbol",
            # strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y=kpi,
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip(kpi, title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


st.title("EDHEC - Time series playground 🧪")

st.markdown(
    """
    [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/gaetanbrison/nlp) <small> app-predictive-analytics 1.0.0 | May 2022</small>""".format(
        img_to_bytes("./images/github.png")
    ),
    unsafe_allow_html=True,
)




start_date = st.date_input(
        "Select start date",
        date(2017, 8, 1),
        min_value=datetime.strptime("2017-08-01", "%Y-%m-%d"),
        max_value=datetime.now(),
    )









list_dataframes = []
for i in range(0,len(symbols)):
    df_data = pdr.get_data_yahoo(symbols[i])
    df_inter = pd.DataFrame(
        {'symbol': [symbols[i]]*len(list(df_data.index)),
        'date': list(df_data.index),
        kpi: list(df_data[kpi])
        })
    list_dataframes.append(df_inter)

df_master = pd.concat(list_dataframes).reset_index(drop=True)
df_master = df_master[df_master['date'] > pd.to_datetime(start_date)]



st.subheader("01 - Show  Time Series visual")





chart = get_chart(df_master)


# Display both charts together
st.altair_chart((chart).interactive(), use_container_width=True)

st.subheader("02 - Show  Dataset")

head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

if head == 'Head':
    st.dataframe(df_master.head(100))
    #

else:
    st.dataframe(df_master.tail(100))



st.subheader("03 - Forecast Prediction")


st.header("Select Symbol to Forecast")
symbol_forecast = st.selectbox("", symbols)


df_data_2 = pdr.get_data_yahoo(symbol_forecast)
df_inter_2 = pd.DataFrame(
        {'symbol': [symbol_forecast]*len(list(df_data_2.index)),
        'date': list(df_data_2.index),
        kpi: list(df_data_2[kpi])
        })


df_inter_3 = df_inter_2[['date', kpi]]
df_inter_3.columns = ['date', kpi]
df_inter_3 = df_inter_3.rename(columns={'date': 'ds', kpi: 'y'})
df_inter_3['ds']= to_datetime(df_inter_3['ds'])
# define the model
model = Prophet()
# fit the model
model.fit(df_inter_3)
# define the period for which we want a prediction
future = list()
for i in range(5, 13):
	date = '2022-%02d' % i
	future.append([date])
future = DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])
# use the model to make a forecast
forecast = model.predict(future)
# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# plot forecast
fig1 = model.plot(forecast)
st.write(fig1)


st.subheader("04 - Forecast Prediction - Actual vs Prediction")

st.write("Predictions values in dataset format")
df_train = df_inter_3[['ds', 'y']].iloc[:150]
df_predict = df_inter_3[['ds']]

# Fitting a Prophet model
model = Prophet()
model.fit(df_train)
forecast = model.predict(df_predict)
st.write(forecast.head())
ax = (df_inter_3.plot(x='ds',y='y',figsize=(20,5),title='Actual Vs Forecast'))
forecast.plot(x='ds',y='yhat',figsize=(20,5),title='Actual vs Forecast', ax=ax)
#fig3 = model.plot(forecast["ds","y","yhat"]) # Prophet's plot method creates a prediction graph
#st.write(fig3)

st.subheader("05 - Model Parameter Decomposition")
# Plotting the forecast components.
fig4 = model.plot_components(forecast)
st.write(fig4)


snippet = f"""

## Import Packages    


import pandas as pd

import time
from datetime import date, datetime

import altair as alt
from fbprophet import Prophet
import streamlit as st
from vega_datasets import data
import pandas_datareader as pdr

## Visualize time series


list_dataframes = []
for i in range(0,len(symbols)):
    df_data = pdr.get_data_yahoo(symbols[i])
    df_inter = pd.DataFrame(
        'symbol': [symbols[i]]*len(list(df_data.index)),
        'date': list(df_data.index),
        kpi: list(df_data[kpi])
        )
    list_dataframes.append(df_inter)

df_master = pd.concat(list_dataframes).reset_index(drop=True)
df_master = df_master[df_master['date'] > pd.to_datetime(start_date)]

chart = get_chart(df_master)
st.altair_chart((chart).interactive(), use_container_width=True)

## Display underlying dataset

head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

if head == 'Head':
    st.dataframe(df_master.head(100))
    #

else:
    st.dataframe(df_master.tail(100))


## Forecast

df_data_2 = pdr.get_data_yahoo(symbols[i])
df_inter_2 = pd.DataFrame(
        'symbol': [symbols[i]]*len(list(df_data_2.index)),
        'date': list(df_data_2.index),
        kpi: list(df_data_2[kpi])
        )


df_inter_3 = df_inter_2[['date', kpi]]
df_inter_3.columns = ['date', kpi]
df_inter_3 = df_inter_3.rename(columns='date': 'ds', kpi: 'y')
df_inter_3['ds']= to_datetime(df_inter_3['ds'])

model = Prophet()

model.fit(df_inter_3)

future = list()
for i in range(5, 13):
	date = '2022-%02d' % i
	future.append([date])
future = DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])

forecast = model.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

fig1 = model.plot(forecast)
st.write(fig1)

"""
code_header_placeholder = st.empty()
snippet_placeholder = st.empty()
code_header_placeholder.subheader(f"**06 - Code**")
snippet_placeholder.code(snippet)







if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### ** 👨🏼‍💻 App Contributors: **")
st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

st.markdown(f"####  Link to Project Website [here]({'https://github.com/gaetanbrison/app-predictive-analytics'}) 🚀 ")



def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made by ",
        link("https://www.edhec.edu/en", "EDHEC - Gaëtan Brison"),
        "👨🏼‍💻"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()

