# rotations
Learning by DIY: rewrite quaternion functions in so3g.

### Interesting resources that I come across while learning this:
- most definitive reference: https://pypi.org/project/pyerfa/
- a javascript ephemeris calculation code: https://github.com/mivion/ephemeris
- an old numerical trick called resummation: https://www.hellenicaworld.com/Science/Mathematics/en/Kahansummationalgorithm.html. It's interesting to know that int in python can be arbitrarily large.
- another interesting thing I learned about jax is that static argument during compilation can be used as a form of caching, because every set of static argument will lead to recompilation of the function if the static argument hasn't been seen before. This can be used for caching expensive computation: if expensive computation is done outside jax during compilation, it won't be recomputed after
the function is compiled but will recompute if a new static arguments show up.
- also learned different timing formats. Interesting to see a lot of the codes in erfa is legacy code written with preserving numerical precision in mind, such as the resummation technique. In principle these techniques are no longer relevant when our ability to work with arbitrary precision. Here's a summary of some conventions read in one of the erfa header files:
  ```
  **  1) The UT1 date dj1+dj2 is a Julian Date, apportioned in any
  **     convenient way between the arguments dj1 and dj2.  For example,
  **     JD(UT1)=2450123.7 could be expressed in any of these ways,
  **     among others:
  **
  **             dj1            dj2
  **
  **         2450123.7           0.0       (JD method)
  **         2451545.0       -1421.3       (J2000 method)
  **         2400000.5       50123.2       (MJD method)
  **         2450123.5           0.2       (date & time method)
  ```
