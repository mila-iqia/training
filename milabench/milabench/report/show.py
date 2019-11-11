
from report.report import show_perf

report_folder = f'/home/user1/mlperf/results_1/power9'
path = '/home/user1/mlperf/results_1'

p9 = show_perf(f'{path}/power9')
r7 = show_perf(f'{path}/radeon7')
mi60 = show_perf(f'{path}/mi60')
mi50_24 = show_perf(f'{path}/mi50_rocm2.4')
mi50_25 = show_perf(f'{path}/mi50_rocm2.5')

print('P9')
print(p9)

print('R7')
print(r7)

print('MI60')
print(mi60)

print('MI50_rocm2.4')
print(mi50_24)

print('MI50_rocm2.5')
print(mi50_25)


