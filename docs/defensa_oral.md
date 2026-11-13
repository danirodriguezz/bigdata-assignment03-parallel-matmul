# Defensa oral — Assignment 3 (multiplicación de matrices paralela, OpenMP)

Guía en español para explicar el trabajo en 5–10 minutos y responder preguntas. Todos los números salen de
`report/generated/numbers.tex` / `results/summary/` (no inventar otros).

## 1. Pregunta de investigación (30 s)
En un portátil con un i7-9750H (6 núcleos físicos, 12 hilos lógicos con Hyper-Threading), ¿cómo cambian el
tiempo, el speedup S(p) = T_seq/T_p y la eficiencia E(p) = S(p)/p de una multiplicación densa por bloques al
variar el número de hilos OpenMP p, y para qué tamaños n compensa paralelizar?

Factor principal: p ∈ {1,2,3,4,5,6,8,10,12,16,24}; segundo factor: n ∈ {256, 512, 1024, 2048, 3072}.
Extras: kernel sin bloques (ikj) y vectorización SIMD a p fijo (extensión opcional).

## 2. Por qué la comparación es justa (1 min) — lo más importante
- **Baseline equivalente**: el baseline secuencial es *el mismo fichero* `src/c/matmul.c` compilado **sin**
  `-fopenmp`. Los `#pragma omp` se ignoran y las macros `THREAD_ID()`/`TEAM_SIZE()` devuelven 0 y 1. Mismo
  algoritmo, mismo orden de bucles, mismo tamaño de bloque (b = 256), mismo tipo (float64), mismo layout
  (row-major), mismos flags (`-O2`) y mismas entradas. Lo único distinto es el runtime de OpenMP.
  `ldd bin/matmul_seq` demuestra que ni siquiera enlaza libgomp.
- **Mismo núcleo**: el secuencial se fija con `taskset -c 0,6` al núcleo físico 0, que es el "place 0" de
  OpenMP. Así T_par(1)/T_seq mide solo la sobrecarga del runtime.
- **Mismas entradas**: ficheros binarios generados una vez (NumPy PCG64, semilla 42, U[-1,1)) con SHA-256.
- **Afinidad fija y documentada**: `OMP_PLACES=cores OMP_PROC_BIND=close` → hilo t en el núcleo t mod 6.
  p ≤ 6: un hilo por núcleo; 7–12: dos hilos comparten núcleo (SMT); 16 y 24: sobresuscripción (más hilos que
  CPUs lógicas). Verificado con `OMP_DISPLAY_AFFINITY` y registrado en cada fila (`threads_actual`, `cpus`).
- **Hilos de librerías controlados**: `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, `OMP_DYNAMIC=false`.
- **Protocolo**: 5 rondas intercaladas y barajadas, k = 150/50/15/10/5 repeticiones medidas según n,
  mediana + IQR, IC 95 % bootstrap para S(p). Lock global de benchmarks (otros agentes en el portátil).
- **Tamaño de bloque elegido por piloto** (b = 256): con b = 64 el secuencial era ~25 % más lento en n = 2048,
  lo que habría *inflado* el speedup. Elegir el bloque que favorece al secuencial es lo honesto.

## 3. El código (para explicarlo línea a línea)
- `matmul_blocked` (matmul.c): una sola región `#pragma omp parallel`. Cada hilo calcula
  `lo = n*t/p`, `hi = n*(t+1)/p` → filas contiguas de C, reparto equilibrado a ±1 fila.
  Pone a cero sus filas y ejecuta los bucles ii → kk → jj sobre bloques de 256, llamando a `tile_update`
  (i → k → j dentro del bloque; el bucle interior `c[j] += aik * b[j]` es de paso unitario).
- **Sin carreras**: cada hilo escribe solo sus filas de C; A y B solo se leen; la barrera implícita al final
  del `parallel` garantiza que C está completa al volver. No hacen falta locks, atómicos ni reducciones.
- **Resultado bit-idéntico**: cada C[i][j] lo calcula un único hilo con la misma secuencia de operaciones
  (k creciente), así que el resultado paralelo es idéntico bit a bit al secuencial para todo p (lo comprueban
  los tests y el checksum de cada repetición).
- `tile_update` es `noinline` a propósito: inline, gcc -O2 sacaba punteros a la pila (spills) → ~30 % más lento
  en el piloto antiguo (`pilot_tile.csv`).
- `matmul_ikj`: `#pragma omp parallel for schedule(static)` sobre filas → mismo reparto contiguo.
- `bench_main.c`: carga entradas y toca C **fuera** del cronómetro; crea el equipo de hilos antes (región de
  prueba); cronometra una llamada al kernel con `CLOCK_MONOTONIC`; calcula el checksum después; imprime una
  línea JSON por repetición con `threads_actual`, CPUs, `cpu_util` y `other_busy_cpus`.

## 4. Corrección
- `make test`: identidad, cero, productos 2×2 y 3×3 exactos, tamaños impares, n < p, n no múltiplo del bloque,
  bloques 1/7/64/200/256, aleatorias vs ijk ingenuo con la cota 2γ_n|A||B| (Higham). Error máximo observado:
  0,64 % de la cota. Bit-identidad paralela para p = 1, 2, 6, 7, 12, 24. Comparación con NumPy.
- ThreadSanitizer: da avisos, pero todos en direcciones de pila (el bloque de variables compartidas que libgomp
  pasa a los hilos), ninguno en `tile_update`. libgomp no está instrumentada → TSan no ve sus barreras →
  falsos positivos conocidos. El argumento fuerte es el de filas disjuntas + bit-identidad.

## 5. Resultados clave (rellenar con numbers.tex)
RESULTADOS

## 6. Limitaciones
- Una sola máquina (portátil) y un solo compilador (gcc 13, libgomp).
- Sin contadores hardware (`perf_event_paranoid=4`): la frecuencia se estima con cpufreq (APERF/MPERF),
  no se miden fallos de caché ni ancho de banda directamente → las explicaciones de memoria son inferencias
  apoyadas en la comparación blocked vs ikj y en el modelo roofline, no mediciones directas.
- Gobernador `powersave` con turbo: la frecuencia depende del número de núcleos activos y de la temperatura.
- Portátil compartido: hay actividad de fondo (navegador, otros agentes); se registra `other_busy_cpus` y hay un
  análisis de sensibilidad.
- k = 5 repeticiones en n = 3072 (presupuesto de tiempo).
- Kernel portable en C, lejos de una BLAS optimizada; las conclusiones son sobre *este* kernel.

## 7. Cómo reproducir un experimento en directo
```bash
make build && make test
# n = 1024, p = 6, 1 warm-up + 3 repeticiones:
OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=false \
  bin/matmul_omp --n 1024 --a data/inputs/n1024_s42_A.bin --b data/inputs/n1024_s42_B.bin \
  --kernel blocked --bs 256 --warmup 1 --reps 3
# baseline secuencial en el mismo núcleo:
taskset -c 0,6 bin/matmul_seq --n 1024 --a data/inputs/n1024_s42_A.bin --b data/inputs/n1024_s42_B.bin \
  --kernel blocked --bs 256 --warmup 1 --reps 3
make analysis   # regenera tablas, figuras y numbers.tex desde results/raw
```

## 8. Preguntas probables y respuestas cortas
PREGUNTAS
