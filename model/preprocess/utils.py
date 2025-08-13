# normalize a value to be a float value if it can be converted/intepretted
def normalize_value(val):
    no_spaces = val.replace(" ", "")
    if val is None:
        return None
    elif "/" in no_spaces:
        fraction = no_spaces.split("/")
        num = float(fraction[0])
        den = float(fraction[1])
        return num / den
    else:
        try:
            return float(no_spaces)
        except ValueError:
            return None