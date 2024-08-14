# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.

import os
import webbrowser
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import panel as pn
from loguru import logger


def df_to_html(
    save_df: pd.DataFrame,
    file_name: str | None = None,
    show: bool = False,
    precision: int = 2,
    **kwargs: dict[Any, Any],
) -> Optional[str]:
    """
    Saves a Pandas DataFrame to an HTML file, optionally showing it in the browser
    :param save_df: the DataFrame to save
    :param file_name: the file name to save to.  If None, will just return the HTML
    :param show: whether to show the HTML in the browser after saving
    :param precision: the number of decimal places to show
    :param kwargs: any other kwargs to pass to Tabulator.  Overrides default values. See https://tabulator.info/docs/5.4/layout and
        https://panel.holoviz.org/reference/widgets/Tabulator.html for details.
    :return: the HTML string, or None if saved to file.
    """
    if len(kwargs) == 0:
        kwargs.update({"layout": "fit_data", "selectable": "checkbox", "header_filters": True})  # type: ignore

    save_df = save_df.copy().round(precision)
    with pd.option_context("display.precision", precision):
        df_widget = pn.widgets.Tabulator(save_df, **kwargs)
        pn.extension("tabulator", css_files=[pn.io.resources.CSS_URLS["font-awesome"]])

        if file_name is not None:
            df_widget.save(Path(file_name).absolute(), embed=True)

            if show:
                url = f"file://{Path(file_name).absolute()}"
                webbrowser.open(url)

        if file_name is None:
            buf = StringIO()
            df_widget.save(buf, embed=True)
            return str(buf.getvalue())
    return None


def figs_to_html(
    figs: list[Any],
    file: str | Path,
    show: bool = False,
    extra_html: str | None = None,
    verbose: int = 0,
    theme: str = "dark",
) -> Path:
    """
    Saves multiple independent Plotly figures to a single HTML page, optionally prepended with an arbitrary html block.
    :param figs: list fo figures to save
    :param file: .html file path to save to
    :param show: whether to show after saving.
    :param extra_html: an arbitrary html block to place above the plot - eg a data table, heading, description etc.
    :param verbose: whether to log the saved file success/location etc.
    :param theme: the Panel theme to use (e.g. 'dark' or 'light').
    :return:
    """
    pn.extension(theme=theme)

    html_file = f"{file}.html"
    parent_dir = Path(html_file).parent

    if not parent_dir.exists():
        parent_dir.mkdir(exist_ok=True, parents=True)

    with open(html_file, "w") as f:
        if extra_html is not None:
            f.write(extra_html)
        for p in figs:
            f.write(p.to_html(full_html=False, include_plotlyjs="cdn"))
        if verbose > 0:
            logger.info(f"Plot saved to {os.path.abspath(html_file)}")
    if show:
        url = "file://" + os.path.abspath(html_file)
        webbrowser.open(url)
    return parent_dir / html_file
