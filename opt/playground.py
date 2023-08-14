from datetime import date, datetime


def conv_str_date(date_str):
    try:
        parts = date_str.split("/")
        day_part = parts[-1].split("(")[0].strip().split("（")[0].strip()
        day = int(day_part)
        month = int(parts[-2])
        year = date.today().year if len(parts) == 2 else int(parts[0])
        return date(year, month, day)

    except (IndexError, ValueError, AttributeError):
        return None


def test_conv_str_date():
    assert conv_str_date("1/1") == date(datetime.now().year, 1, 1)
    assert conv_str_date("1/1/1") == date(1, 1, 1)
    assert conv_str_date("1/1/1（月）") == date(1, 1, 1)
    assert conv_str_date("1/1（月）") == date(datetime.now().year, 1, 1)
    assert conv_str_date("1/1(月)") == date(datetime.now().year, 1, 1)


test_conv_str_date()
