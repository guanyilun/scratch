"""Convert various time formats"""
#%%
from typing import NamedTuple
import numpy as np

class UTC(NamedTuple):
    v1: float
    v2: float

class UT1(NamedTuple):
    v1: float
    v2: float

class JD(NamedTuple):
    v1: float
    v2: float

class TAI(NamedTuple):
    v1: float
    v2: float

class TT(NamedTuple):
    v1: float
    v2: float

class Gregorian(NamedTuple):
    year: int
    month: int
    day: int
    frac_day: float

JDMIN = -68569.5
DJMAX = 1.0e9
def jd2cal(jd: JD) -> Gregorian:
    """
    Julian Date to Gregorian Calendar
    """
    dj = jd.v1 + jd.v2
    assert dj >= JDMIN and dj < DJMAX, "jd2cal: invalid date"
    # separate day and fraction
    d  = round(jd.v1)
    f1 = jd.v1 - d
    jd = d
    d  = round(jd.v2)
    f2 = jd.v2 - d
    jd += d
    # compute f1+f2+0.5 using compensated summation (Klein 2006):
    # https://www.hellenicaworld.com/Science/Mathematics/en/Kahansummationalgorithm.html
    s, cs = 0.5, 0
    for x in [f1, f2]:
        t = s + x
        if abs(s) >= abs(x):
            cs += (s - t) + x
        else:
            cs += (x - t) + s
        s = t
        if s >= 1:
            jd += 1
            s -= 1
    f = s + cs
    cs = f - s
    if f < 0:
        f = s + 1.
        cs += (1.-f) + s
        s = f
        f = s + cs
        cs = f - s
        jd -= 1
    if f >= 1:
        t = s - 1.
        cs += (s-t) - 1.
        s = t
        f = s + cs
        if f > 0:
            jd += 1
            f = max(f, 0)
    # the above code outputs: f, jd

    l = jd + 68569
    n = 4*l // 146097
    l -= (146097*n + 3) // 4
    i = 4000*(l+1) // 1461001
    l -= (1461*i) // 4 - 31
    k = 80*l // 2447
    day = int(l - (2447*k) // 80)
    l = k // 11
    month = int(k + 2 - 12*l)
    year = int(100*(n-49) + i + l)
    return Gregorian(year, month, day, f)

def cal2jd(cal: Gregorian) -> JD:
    """
    Gregorian Calendar to Julian Date
    """
    year, month, day, _ = cal
    assert year >= -4712, "cal2jd: invalid date"
    assert month >= 1 and month <= 12, "cal2jd: invalid date"

    DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    leap_year = (month == 2) and (not (year % 4)) and ((year % 100) or ~(year % 400))
    assert day >= 1 and day <= DAYS_IN_MONTH[month-1] + leap_year, "cal2jd: invalid date"

    my = (month - 14) // 12
    jd0 = 2400000.5
    jd1 = 1461 * (year + 4800 + my) // 4 + 367 * (month - 2 - 12*my) // 12 - \
        (3 * ((year + 4900 + my) // 100)) // 4 + day - 24320762
    return JD(jd0, jd1)

# erfa: eraLEAPSECONDS
# iyear, month, delat
LEAP_SECONDS = np.array([
    [1960, 1, 1.4178180],
    [1961, 1, 1.4228180],
    [1961, 8, 1.3728180],
    [1962, 1, 1.8458580],
    [1963, 11, 1.9458580],
    [1964, 1, 3.2401300],
    [1964, 4, 3.3401300],
    [1964, 9, 3.4401300],
    [1965, 1, 3.5401300],
    [1965, 3, 3.6401300],
    [1965, 7, 3.7401300],
    [1965, 9, 3.8401300],
    [1966, 1, 4.3131700],
    [1968, 2, 4.2131700],
    [1972, 1, 10.0],
    [1972, 7, 11.0],
    [1973, 1, 12.0],
    [1974, 1, 13.0],
    [1975, 1, 14.0],
    [1976, 1, 15.0],
    [1977, 1, 16.0],
    [1978, 1, 17.0],
    [1979, 1, 18.0],
    [1980, 1, 19.0],
    [1981, 7, 20.0],
    [1982, 7, 21.0],
    [1983, 7, 22.0],
    [1985, 7, 23.0],
    [1988, 1, 24.0],
    [1990, 1, 25.0],
    [1991, 1, 26.0],
    [1992, 7, 27.0],
    [1993, 7, 28.0],
    [1994, 7, 29.0],
    [1996, 1, 30.0],
    [1997, 7, 31.0],
    [1999, 1, 32.0],
    [2006, 1, 33.0],
    [2009, 1, 34.0],
    [2012, 7, 35.0],
    [2015, 7, 36.0],
    [2017, 1, 37.0]
])

# erfa: dat.c
# Reference dates (MJD) and drift rates (s/day), pre leap seconds
DRIFTS = np.array([
   [37300.0, 0.0012960],
   [37300.0, 0.0012960],
   [37300.0, 0.0012960],
   [37665.0, 0.0011232],
   [37665.0, 0.0011232],
   [38761.0, 0.0012960],
   [38761.0, 0.0012960],
   [38761.0, 0.0012960],
   [38761.0, 0.0012960],
   [38761.0, 0.0012960],
   [38761.0, 0.0012960],
   [39126.0, 0.0025920],
   [39126.0, 0.0025920]
])


def dt_tai_utc(cal: Gregorian) -> float:
    """
    Difference between TAI and UTC in seconds: delta = TAI - UTC
    """
    assert cal.year >= LEAP_SECONDS[0].year, "dt_tai_utc: older than 1960 not supported"
    assert cal.year <= 2028, "dt_tai_utc: too far in the future not supported"
    m = cal.year * 12 + cal.month
    # find preceding leap second entry
    i_preceding = np.where(LEAP_SECONDS[:, 0]*12+LEAP_SECONDS[:, 1] <= m)[0][-1]
    delta = LEAP_SECONDS[i_preceding, -1]
    # adjust for drift in pre-1972 entries
    if i_preceding < DRIFTS.shape[0]:
        jd = cal2jd(cal)
        delta += (jd.v1 + jd.v2 - DRIFTS[i_preceding, 0]) * DRIFTS[i_preceding, 1]
    return delta

class TAI(NamedTuple):
    v1: float
    v2: float

# erfa: utctai.c
def utc2tai(utc: UTC) -> TAI:
    u1, u2 = utc.v1, utc.v2
    if abs(u1) < abs(u2): u1, u2 = u2, u1
    jd = JD(u1, u2)
    cal = jd2cal(jd)
    # 0h today
    delta_tai_utc_0h  = dt_tai_utc(cal._replace(frac_day=0))
    # 12h later: to detect drift
    delta_tai_utc_12h = dt_tai_utc(cal._replace(frac_day=0.5))
    # 0h next day: to detect jumps
    jd = JD(jd.v1+1.5, jd.v2-cal.frac_day)
    cal = jd2cal(jd)
    delta_tai_utc_24h = dt_tai_utc(cal._replace(frac_day=0))
    # separate TAI-UTC change into per-day (DLOD) and any jump (DLEAP)
    dlod = 2. * (delta_tai_utc_12h - delta_tai_utc_0h)
    dleap = delta_tai_utc_24h - delta_tai_utc_0h - dlod
    # remove scaling applied to spread leap into preceding day
    f = cal.frac_day
    f *= (86400.0 + dlod) / 86400.0
    f *= (86400.0 + dleap) / 86400.0
    # today's calendar date to JD
    jd = cal2jd(cal)
    a2 = jd.v1 - u1 + jd.v2 + f + delta_tai_utc_0h/86400.0
    if utc.v1 < utc.v2: u1, a2 = a2, u1
    return TAI(u1, a2)

# erfa: taiutc.c
def tai2ut1(tai: TAI, dta: float) -> UT1:
    """
    dta: UT1-TAI in seconds
    """
    dta_day = dta / 86400.0
    if abs(tai.v1) > abs(tai.v2):
        return UT1(tai.v1, tai.v2 + dta_day)
    else:
        return UT1(tai.v1 + dta_day, tai.v2)

# erfa: utcut1.c
def utc2ut1(utc: UTC, dut1: float) -> UT1:
    """
    dut1: UT1-UTC in seconds
    """
    cal = jd2cal(JD(utc.v1, utc.v2))
    delta_tai_utc = dt_tai_utc(cal)
    # UT1-TAI = UT1-UTC-(TAI-UTC) = dut1 - delta_tai_utc
    dta = dut1 - delta_tai_utc
    return tai2ut1(utc2tai(utc), dta)

def tai2tt(tai: TAI) -> TT:
    """International Atomic Time, TAI, to Terrestrial Time, TT."""
    dtaitt = 32.184 / 86400.0
    if abs(tai.v1) > abs(tai.v2):
        return TT(tai.v1, tai.v2 + dtaitt)
    else:
        return TT(tai.v1 + dtaitt, tai.v2)

def ctime2jd(ctime: float) -> JD:
    """
    Convert unix time to Julian Date
    """
    return JD(2440587.5, ctime/86400.0)

ERA_POLY = [0.00273781191135448, 0.7790572732640]
def era(ut1: UT1) -> float:
    """Earth Rotation Angle in radians"""
    if ut1.v1 < ut1.v2: d1, d2 = ut1.v1, ut1.v2
    else: d1, d2 = ut1.v2, ut1.v1
    t = d1 + (d2 - 2451545.0)
    # Fractional part of T (days).
    f = d1 % 1 + d2 % 1
    # Earth rotation angle at this UT1.
    theta = (f + np.polyval(ERA_POLY, t)) % 1
    return theta * 2 * np.pi