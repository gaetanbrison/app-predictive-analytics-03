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

st.subheader("Show  Time Series visual")

chart = get_chart(df_master)


# Display both charts together
st.altair_chart((chart).interactive(), use_container_width=True)

st.subheader("Show  Dataset")

head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

if head == 'Head':
    st.dataframe(df_master.head(100))
    #

else:
    st.dataframe(df_master.tail(100))

snippet = f"""

        >>> import pandas as pd
        >>> import numpy as  as np
        >>> import nltk

        >>> doc = nlp(df["Review"][0])
        >>> list_text = []
        >>> list_pos = []
        >>> list_lemma = []
        >>> list_lemma_ = []
        >>> for token in doc:
            >>> list_text.append(token.text)
            >>> list_pos.append(token.pos_)
            >>> list_lemma.append(token.lemma)
            >>> list_lemma_.append(token.lemma_)
        >>> df_lemmatization = pd.DataFrame('Text': list_text, 'Position': list_pos, 'Unique Code': list_lemma)
        >>> df_lemmatization

        """
code_header_placeholder = st.empty()
snippet_placeholder = st.empty()
code_header_placeholder.subheader(f"**Code for the step: 03 - Lemmatization**")
snippet_placeholder.code(snippet)







if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### ** 👨🏼‍💻 App Contributors: **")
st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

st.markdown(f"####  Link to Project Website [here]({'https://github.com/gaetanbrison'}) 🚀 ")



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

