A naive python implementation of wigner3j_f function in WignerFamilies.jl

```python
t0 = time.time()
j1_vals, symbols = wigner3j_f(100, 60, 70, -55)
t1 = time.time()
print(f"Time taken: {t1 - t0:.4f} seconds")

plt.plot(j1_vals, symbols*1000)
plt.xlabel('j1')
plt.ylabel('Wigner 3j symbol x 1000')
```

gives
```
Time taken: 0.0002 seconds
```
and the plot is consistent with julia implementation:

![image](https://github.com/user-attachments/assets/6080ace9-e09b-4359-a1b6-a5d09c8a8580)
