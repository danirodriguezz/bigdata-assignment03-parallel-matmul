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
  (row-major), mismos flags (`-O2 -falign-loops=32`) y mismas entradas. Lo único distinto es el runtime de OpenMP.
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
Todos de run 3 (todas las variantes con `-falign-loops=32`). Medianas; IC 95 % bootstrap.

1. **Paralelizar compensa en todos los tamaños**, incluso n = 256 (kernel de 9 ms). Con 6 hilos (uno por núcleo
   físico): S(6) = 4,68–4,99 para n ≥ 512 (E ≈ 0,78–0,83); en n = 3072: T_seq = 16,7 s → T_6 = 3,49 s,
   S(6) = 4,79 [3,93; 5,80]. 6 hilos queda como mucho un 7 % por debajo de la mejor configuración en cada n.
2. **La eficiencia ≈ 0,8 se explica por la frecuencia**: 3770 MHz con 1 hilo → 2900 MHz con 6 (f(6)/f(1) = 0,77).
   Cota "solo frecuencia" 6·0,77 = 4,61; speedup relativo medido T_par(1)/T_6 = 4,54 (n = 2048) y 4,72 (n = 3072).
   → por ciclo, el código paralelo escala casi perfecto. (Frecuencia de cpufreq en ejecuciones aparte, no contadores.)
3. **SMT y sobresuscripción no aportan**: S(12)/S(6) = 0,97–1,03; S(24)/S(12) = 0,85–1,07. **Bajón en p = 8**
   (S(8)/S(6) = 0,74–0,83) por reparto desigual hilos/núcleo (2,2,1,1,1,1) + barrera; cpu_util = 6,56 de 8.
   Modelo simple T_p/T_6 ≈ m·6/p (m = hilos en el núcleo más cargado) predice 0,67 / 0,83 / 1 para p = 8/10/12
   y S(16)/S(24) ≈ 0,89: acierta el orden, exagera algo la pérdida.
4. **Sobrecarga a 1 hilo no medible** tras alinear: T_par(1)/T_seq entre −7,2 % y +2,3 % (IC contiene 1 salvo n = 256,
   donde el paralelo es *más rápido*). Sin alinear (runs 1 y 2): +19 % a +27 % → artefacto de colocación de código.
5. **Memoria**: RSS ≈ almacenamiento de las 3 matrices (n = 3072: 217,5 MiB seq vs 216 MiB teórico); OpenMP
   +0,3 MiB (libgomp) y prácticamente nada por hilo (≤ 0,14 MiB entre p).
6. **ikj vs bloques**: en speedup el ikj escala igual o mejor (S(6) = 5,16 vs 4,68 en n = 2048) porque su baseline es
   más lento; en GFLOP/s el de bloques gana (16,6 vs 12,3) y el ikj se hunde con p = 24 (8,0 GFLOP/s).
7. **Vectorización** (a p fijo): AVX2 ×2,27–2,30 con p = 1 y ×1,98–2,15 con p = 6; FMA algo más con p = 1
   (×2,46–2,52), indistinguible con p = 6. Mejor: AVX2+FMA, p = 6, n = 2048 → 36,9 GFLOP/s ≈ 10× el escalar
   secuencial (3,4 GFLOP/s).
8. **Dos días**: rondas 1–2 el 13/11 (Linux 7.0.0-42), rondas 3–5 el 16/11 (7.0.0-43, más carga de fondo:
   0,23 → 0,92 CPUs ajenas). El secuencial fue 10–21 % más lento el día 2, pero como cada ronda contiene todas las
   configuraciones, el speedup por ronda (pareado) difiere del agregado solo un 2,3 % de mediana (máx. 10,4 %).

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
**¿Por qué no comparas con el binario paralelo a 1 hilo en vez de con el secuencial?**
Porque el enunciado exige el baseline secuencial equivalente y porque T_par(1) incluye la sobrecarga del runtime;
S(p) respecto a T_par(1) (speedup "relativo") sobreestima la ganancia real. Reporto ambos: S(p) con T_seq como
resultado principal y T_par(1)/T_seq como medida de sobrecarga.

**¿Qué pasó con la primera ejecución (run 1)?**
El binario OpenMP salía ~20 % más lento que el secuencial *con cualquier número de hilos*, aunque el bucle interno
es idéntico instrucción a instrucción. La variante `omp_scalar` (mismo código máquina del kernel) igualaba al
secuencial. Mirando el desensamblado de los binarios enlazados (`results/evidence/loop_placement.csv`): los casos
lentos son justo aquellos cuyo bucle interno cruza una frontera de línea de caché de 64 bytes; su dirección depende
del tamaño de código que no tiene nada que ver (p. ej. `bench_main.c`). Con `-falign-loops=32` (bucle de 30 bytes
alineado a 32 → nunca cruza 64 B) la diferencia desaparece; repetí todo el benchmark con ese flag en todas las
variantes. Los datos de run 1 se conservan sin tocar. **Run 2** iba a ser la ejecución alineada, pero `make build` no
recompiló (los binarios no dependían del Makefile; las filas registran los flags antiguos): es una **réplica
independiente** de run 1 con el mismo efecto (+21 a +27 % en el kernel por bloques), lo que refuerza que es reproducible. Arreglado el Makefile,
**run 3** (binarios con `-falign-loops=32`, verificado en `build_flags` de cada fila) da todos los resultados. Es el clásico "producing wrong data without doing anything
obviously wrong" (Mytkowicz et al. 2009). El mecanismo exacto del front-end es conjetura (sin contadores hardware).

**¿Por qué cae el speedup en p = 8 y vuelve a subir en p = 12?**
Con `OMP_PLACES=cores` y `close`, con 8 hilos los núcleos 0 y 1 tienen 2 hilos cada uno y los otros 4 núcleos uno.
El reparto de filas es igual para todos los hilos, así que los 4 hilos que comparten núcleo van más lentos y el
resto espera en la barrera final: desequilibrio de carga. Con 12 hilos todos los núcleos tienen 2 hilos → vuelve a
estar equilibrado y el tiempo ≈ p = 6 (el SMT casi no aporta a este kernel escalar). Lo mismo con 16 (desequilibrado)
vs 24 (equilibrado). `cpu_util` lo confirma: con p = 8 la CPU usada es 6,56, no 8.

**¿Por qué la eficiencia con 6 hilos no es 1?**
Principalmente por la frecuencia: con 1 núcleo activo ~3,77 GHz y con 6 núcleos ~2,90 GHz (freq_probe).
Si solo limitase la frecuencia, S(6) ≈ 6 · 0,77 = 4,61, y el speedup relativo medido es 4,54–4,72.

**¿Hay speedup superlineal?**
En los valores agregados no (0 casos con S > p). Por ronda sí: 25 de 125 celdas (p = 2–6) tienen S > p (máx.
E = 1,21 en n = 256, p = 2). Lo investigué: en esas rondas el proceso *secuencial* fue ~20 % más lento que el paralelo
a 1 hilo de la misma ronda, coincidiendo con picos de actividad ajena (p. ej. 2,3–2,5 CPUs ajenas en n = 256). El
speedup relativo T_par(1)/T_p supera p solo en 3 de esas 25 celdas y por poco → es ruido del baseline, no
superlinealidad real.

**¿Cómo sabes que la vectorización ocurre?**
Tres evidencias: (1) informe del compilador `-fopt-info-vec-optimized` (bucle `c[j] += aik*b[j]`, matmul.c:40,
"loop vectorized using 32 byte vectors"); (2) desensamblado: la variante escalar solo tiene `mulsd/addsd`, la AVX2
tiene `vmulpd/vaddpd` sobre registros `%ymm` y la FMA `vfmadd...pd`; (3) el tiempo baja ~2× a p fijo.
`-O2` sin `-fno-tree-vectorize` genera *el mismo* `.text` que con él (gcc 13 a -O2 usa el modelo de coste
"very-cheap" y no vectoriza este bucle), así que el barrido principal es escalar.

**¿Cómo separas vectorización de hilos?**
Comparo variantes SIMD *al mismo p* (1 y 6): speedup SIMD = T_scalar(p)/T_variant(p). Y el speedup por hilos de
cada variante (T(1)/T(6)) con el mismo binario.

**¿Por qué FMA da resultados distintos?**
FMA redondea una vez en vez de dos (a·b+c), así que cambia los bits del resultado (sigue dentro de la cota de
error). Por eso con `-std=c11` gcc no contrae a FMA salvo `-ffp-contract=fast`, y lo trato como variante aparte.

**¿Por qué b = 256 si con n pequeño b = 64 era algo mejor?**
Con b = 64 el secuencial era ~25 % más lento en n = 2048; un baseline peor infla el speedup. b = 256 tiene la menor
pérdida máxima y coincide con el elegido en el Assignment 2.

**¿Qué es `other_busy_cpus`?**
CPU consumida por *otros* procesos durante cada repetición (de /proc/stat). Sirve para detectar interferencias
(navegador, otros agentes). El análisis de sensibilidad 1 excluye repeticiones con > 1 CPU ajena (poco fiable en n grande: deja k = 1–2);
el 2 (mejor) calcula S dentro de cada ronda y toma la mediana de las 5 rondas.

**¿Por qué las rondas 3–5 son de otro día? ¿No invalida eso los resultados?**
Pausé la ejecución el 13/11 (apagado del portátil) y la reanudé el 16/11 con `--resume`. Los tiempos absolutos suben
(kernel nuevo, más carga de fondo), pero cada ronda contiene todas las configuraciones en orden aleatorio, así que el
cociente seq/par se mide en las mismas condiciones. El análisis pareado por ronda lo confirma (diferencia mediana 2,3 %).
Lo declaro en Resultados y Limitaciones; el coste es que los IC en n grande son anchos.

**¿Qué cambiarías con más tiempo?**
Contadores hardware (fallos de caché, ancho de banda), fijar la frecuencia (desactivar turbo) para separar el
efecto frecuencia, reparto de filas proporcional para p no múltiplo de 6, `schedule(dynamic)`, y comparar con
una BLAS multihilo (OpenBLAS) como referencia de rendimiento máximo.
