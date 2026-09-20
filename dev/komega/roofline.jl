# Memory-bandwidth roof on this machine: how much does streaming scale from 1 to N threads?
# Arrays are far larger than the 36 MB L3 so every access goes to DRAM.
using Base.Threads, Printf, ThreadPinning
pinthreads(:cores)
const n = 40_000_000                  # 320 MB per Float64 array
a = fill(1.0, n); b = fill(2.0, n); c = fill(3.0, n)

function triad_serial!(a, b, c, s)
    @inbounds @simd for i in eachindex(a); a[i] = b[i] + s*c[i]; end
end
function triad_threaded!(a, b, c, s)
    @threads :static for i in eachindex(a)
        @inbounds a[i] = b[i] + s*c[i]
    end
end
best(f, reps=6) = minimum(begin; t=@elapsed f(); t; end for _ in 1:reps)
gbs(t) = 3*n*8/t/1e9                  # 2 reads + 1 write

triad_serial!(a,b,c,1.5); triad_threaded!(a,b,c,1.5)
t1 = best(() -> triad_serial!(a,b,c,1.5))
tn = best(() -> triad_threaded!(a,b,c,1.5))
@printf("threads=%d\n", nthreads())
@printf("triad serial   %7.2f GB/s\n", gbs(t1))
@printf("triad %d-thread %7.2f GB/s   scaling %.2fx\n", nthreads(), gbs(tn), t1/tn)
