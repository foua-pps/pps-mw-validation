from pathlib import Path
import datetime as dt
import pytest  # type: ignore

from pps_mw_validation import ici


@pytest.mark.parametrize("date,expect", (
    (dt.date(2007, 9, 11), False),
    (dt.date(2007, 9, 12), True),
    (dt.date(2007, 9, 13), False),
))
def test_cover_date(
    date: dt.date,
    expect: bool,
):
    filepattern = "ICI_(?P<created>\d+)_G_D_(?P<start>\d+)_(?P<end>\d+)_"  # noqa
    filename = Path("ICI_20230131102850_G_D_20070912084321_20070912102224_.nc")
    assert ici.cover_date(filename, filepattern, date) == expect
