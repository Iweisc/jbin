CC       ?= cc
CXX      ?= c++
CFLAGS    = -Wall -Wextra -Wpedantic -std=c11 -O2
BENCHFLAGS = -O3 -march=native -std=c11 -fno-stack-protector -fno-asynchronous-unwind-tables

ARENA     = -DJBIN_MAX_NODES='(1<<24)' \
            -DJBIN_MAX_STRING='(256u*1024u*1024u)' \
            -DJBIN_MAX_STRUCTURAL='(1<<25)' \
            -D_POSIX_C_SOURCE=199309L

DATA      = data/canada.json data/twitter.json data/citm_catalog.json data/large.json

all: test_runner

test_runner: test_runner.c jbin.c jbin.h
	$(CC) $(CFLAGS) -o $@ test_runner.c jbin.c

bench: bench.c jbin.c jbin.h
	$(CC) $(BENCHFLAGS) $(ARENA) -o $@ bench.c jbin.c -lm

bench-pgo: bench.c jbin.c jbin.h
	$(CC) $(BENCHFLAGS) $(ARENA) -fprofile-generate -o bench bench.c jbin.c -lm
	@echo "Training PGO profile..."
	@for i in 1 2 3 4 5; do ./bench data/twitter.json data/citm_catalog.json >/dev/null; done
	@./bench data/canada.json data/large.json >/dev/null
	@for i in 1 2 3; do ./bench data/numbers.json data/mesh.json >/dev/null 2>/dev/null || true; done
	$(CC) $(BENCHFLAGS) $(ARENA) -fprofile-use -flto -o bench bench.c jbin.c -lm
	@rm -f *.gcda
	@echo "PGO build complete. Run: ./bench -n100"

bench-simdjson: bench_simdjson.cpp simdjson/simdjson.h simdjson/simdjson.cpp
	$(CXX) -O3 -march=native -o $@ bench_simdjson.cpp simdjson/simdjson.cpp

bench-yyjson: bench_yyjson.c yyjson/yyjson.h yyjson/yyjson.c
	$(CC) $(BENCHFLAGS) -D_POSIX_C_SOURCE=199309L -o $@ bench_yyjson.c yyjson/yyjson.c -lm

bench-rapidjson: bench_rapidjson.cpp
	$(CXX) -O3 -march=native -Irapidjson/include -o $@ bench_rapidjson.cpp

bench-cjson: bench_cjson.c cjson/cJSON.h cjson/cJSON.c
	$(CC) $(BENCHFLAGS) -D_POSIX_C_SOURCE=199309L -o $@ bench_cjson.c cjson/cJSON.c -lm

bench-sajson: bench_sajson.cpp sajson/sajson.h
	$(CXX) -O3 -march=native -o $@ bench_sajson.cpp

check: test_runner
	./test_runner

check-v: test_runner
	./test_runner -v

check-suite: test_runner | JSONTestSuite
	./test_runner JSONTestSuite/test_parsing

check-suite-v: test_runner | JSONTestSuite
	./test_runner -v JSONTestSuite/test_parsing

JSONTestSuite:
	git clone --depth 1 https://github.com/nst/JSONTestSuite.git

freestanding-check: jbin.c jbin.h
	$(CC) -std=c11 -ffreestanding -c -o /dev/null jbin.c
	@echo "freestanding compilation OK"

clean:
	rm -f test_runner bench bench-simdjson bench-yyjson bench-rapidjson bench-cjson bench-sajson *.o *.gcda

.PHONY: all check check-v check-suite check-suite-v freestanding-check clean bench-pgo
