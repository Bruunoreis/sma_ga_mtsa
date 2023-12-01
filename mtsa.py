import numpy as np
from mealpy import PermutationVar, SMA, Problem
import matplotlib.pyplot as plt

city_positions = np.array([
    [0, 400, 1000, 1600, 850, 3600, 2500, 2600, 1000, 2300, 1800],
    [400, 0, 900, 1500, 600, 3300, 2700, 2300, 1200, 2400, 1700],
    [1000, 900, 0, 900, 1800, 3800, 3600, 2800, 1800, 2000, 850],
    [1600, 1500, 900, 0, 2000, 4100, 3400, 3100, 2400, 1600, 800],
    [850, 600, 1800, 2000, 0, 3600, 2200, 2600, 1200, 3300, 2200],
    [3600, 3300, 3800, 4100, 3600, 0, 2200, 1700, 2800, 5500, 4600],
    [2500, 2700, 3600, 3400, 2200, 1700, 0, 2400, 1100, 3700, 3900],
    [2600, 2300, 2800, 3100, 2600, 2800, 2400, 0, 3300, 6000, 4100],
    [1000, 1200, 1800, 2400, 1200, 5500, 1100, 3300, 0, 2900, 3100],
    [2300, 2400, 2000, 1600, 3300, 5500, 3700, 6000, 2900, 0, 2000],
    [1800, 1700, 850, 900, 2200, 4600, 3900, 4100, 3100, 2000, 0]])
num_cities = len(city_positions)
data = {
    "city_positions": city_positions,
    "num_cities": num_cities,
}

class MultipleTravelingSalesmenProblem(Problem):
    def _init_(self, num_salesmen, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        self.num_salesmen = num_salesmen
        self.eps = 1e10  # Penalidade para vértice com conexão zero
        super()._init_(bounds=bounds, minmax=minmax, **kwargs)

    def decode_solution(self, x):
        paths_decodificados = {}
        num_nodes = len(self.data)

        # Determine o número de cidades que cada caixeiro deve visitar
        cidades_por_caixeiro = num_nodes // self.num_salesmen

        # Decode a solução para cada caixeiro
        for i in range(self.num_salesmen):
            start_idx = i * cidades_por_caixeiro
            end_idx = start_idx + cidades_por_caixeiro
            caminho_caixeiro = [0]+[int(idx) for idx in x[start_idx:end_idx] if idx != 0] + [0]
            paths_decodificados[f"caixeiro_{i+1}"] = caminho_caixeiro

        return paths_decodificados
 

    def obj_func(self, x):
        paths_decodificados = self.decode_solution(x)
        distancias_totais = []

        for caminho_caixeiro in paths_decodificados.values():
            distancia_total = 0
            for idx in range(len(caminho_caixeiro) - 1):
                no_inicial = caminho_caixeiro[idx]
                no_final = caminho_caixeiro[idx + 1]
                peso = self.data[no_inicial, no_final]
                if peso == 0:
                    return self.eps
                distancia_total += peso
            distancias_totais.append(distancia_total)

        return sum(distancias_totais)

num_salesmen = 3

cidades_por_caixeiro = 20  # Número de caixeiros viajantes
num_nodes = len(city_positions)
bounds = [PermutationVar(valid_set=list(range(0, num_nodes)), name=f"path_{i}") for i in range(num_salesmen)]
problem = MultipleTravelingSalesmenProblem(num_salesmen, bounds=bounds, minmax="min", data=city_positions)

model_mtsp = SMA.OriginalSMA(epoch=100, pop_size=10)
model_mtsp.solve(problem)

print(f"Best agent: {model_mtsp.g_best}")                    # Encoded solution
print(f"Best solution: {model_mtsp.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model_mtsp.g_best.target.fitness}")
print(f"Best real scheduling: {model_mtsp.problem.decode_solution(model_mtsp.g_best.solution)}")

best_solution = model_mtsp.g_best.solution
decoded_solution = problem.decode_solution(best_solution)

# Função para plotar a rota de todos os caixeiros juntos
def plot_routes(routes, city_positions):
    plt.figure()
    plt.title('Rotas de Todos os Caixeiros')

    # Cores diferentes para cada caixeiro
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(routes))))

    for caixeiro, rota in routes.items():
        x = [city_positions[0, i] for i in rota]
        y = [city_positions[1, i] for i in rota]
        x.append(x[0])  # Adiciona o ponto inicial no final para fechar o ciclo
        y.append(y[0])
        
        # Obtém a próxima cor da paleta
        color = next(colors)
        
        plt.plot(x, y, marker='o', label=f'Rota do {caixeiro}', color=color)
        for i, txt in enumerate(rota):
            plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.legend()
    plt.legend()
    plt.show()

# Plotar a rota para todos os caixeiros
plot_routes(decoded_solution, city_positions)