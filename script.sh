#!/bin/bash

# Compila os arquivos .cu usando nvcc
./compila.sh

# Executa os programas compilados
echo "Executando aquecimento"

./mppSort 1000000 1024 1

echo "Executando teste 1M"

./mppSort 1000000 1024 10

echo "Executando teste 2M"

./mppSort 2000000 1024 10

echo "Executando teste 4M"

./mppSort 4000000 1024 10

echo "Executando teste 8M"

./mppSort 8000000 1024 10

