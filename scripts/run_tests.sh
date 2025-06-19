#!/bin/bash

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar mensajes
print_message() {
    echo -e "${2}${1}${NC}"
}

# Función para mostrar el menú de ayuda
show_help() {
    echo "Uso: ./run_tests.sh [opciones]"
    echo ""
    echo "Opciones:"
    echo "  -h, --help              Mostrar este mensaje de ayuda"
    echo "  -a, --all               Ejecutar todos los tests"
    echo "  -u, --unit              Ejecutar solo tests unitarios"
    echo "  -o, --optimization      Ejecutar solo tests de optimización"
    echo "  -c, --coverage          Ejecutar tests con cobertura"
    echo "  -b, --benchmark         Ejecutar tests de rendimiento"
    echo "  -v, --verbose           Mostrar output detallado"
    echo ""
    echo "Ejemplos:"
    echo "  ./run_tests.sh -a        # Ejecutar todos los tests"
    echo "  ./run_tests.sh -u -v     # Ejecutar tests unitarios con output detallado"
    echo "  ./run_tests.sh -o -c     # Ejecutar tests de optimización con cobertura"
}

# Variables por defecto
RUN_ALL=false
RUN_UNIT=false
RUN_OPTIMIZATION=false
RUN_COVERAGE=false
RUN_BENCHMARK=false
VERBOSE=false

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -u|--unit)
            RUN_UNIT=true
            shift
            ;;
        -o|--optimization)
            RUN_OPTIMIZATION=true
            shift
            ;;
        -c|--coverage)
            RUN_COVERAGE=true
            shift
            ;;
        -b|--benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_message "Opción desconocida: $1" "$RED"
            show_help
            exit 1
            ;;
    esac
done

# Si no se especificó ninguna opción, mostrar ayuda
if [ "$RUN_ALL" = false ] && [ "$RUN_UNIT" = false ] && [ "$RUN_OPTIMIZATION" = false ]; then
    show_help
    exit 1
fi

# Función para ejecutar tests
run_tests() {
    local test_type=$1
    local coverage=$2
    local benchmark=$3
    
    print_message "Ejecutando $test_type tests..." "$YELLOW"
    
    # Construir comando pytest
    cmd="pytest"
    
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi
    
    if [ "$coverage" = true ]; then
        cmd="$cmd --cov=hyperion3 --cov-report=term-missing"
    fi
    
    if [ "$benchmark" = true ]; then
        cmd="$cmd --benchmark-only"
    fi
    
    # Añadir directorio de tests específico
    case $test_type in
        "unit")
            cmd="$cmd tests/test_*.py -k 'not TestOptimization'"
            ;;
        "optimization")
            cmd="$cmd tests/test_optimization.py"
            ;;
        "all")
            cmd="$cmd tests/"
            ;;
    esac
    
    # Ejecutar tests
    eval $cmd
    
    # Verificar resultado
    if [ $? -eq 0 ]; then
        print_message "✅ Tests $test_type completados exitosamente" "$GREEN"
    else
        print_message "❌ Tests $test_type fallaron" "$RED"
        exit 1
    fi
}

# Ejecutar tests según las opciones seleccionadas
if [ "$RUN_ALL" = true ]; then
    run_tests "all" "$RUN_COVERAGE" "$RUN_BENCHMARK"
elif [ "$RUN_UNIT" = true ]; then
    run_tests "unit" "$RUN_COVERAGE" "$RUN_BENCHMARK"
elif [ "$RUN_OPTIMIZATION" = true ]; then
    run_tests "optimization" "$RUN_COVERAGE" "$RUN_BENCHMARK"
fi

print_message "✨ Todos los tests completados" "$GREEN" 