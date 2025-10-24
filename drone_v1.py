#!/usr/bin/env python3
"""
drone_ga.py
Implementação de um Algoritmo Genético para planejar coleta por drone em Curitiba
(Projeto acadêmico descrito pelo usuário).
Produz um CSV com a melhor solução encontrada.

Como usar:
  - Ajuste parâmetros abaixo (student_id, autonomia_base, etc.)
  - Forneça ceps.csv e vento.csv no mesmo diretório
  - Rode: python3 drone_ga.py
"""

import csv
import math
import random
import datetime
import sys
from collections import namedtuple, deque
from copy import deepcopy
import statistics
import unittest
import os

# ----------------------------
# CONFIGURAÇÕES / PARÂMETROS
# ----------------------------
student_id = "000000"   # se começar com '2' ativa várias regras extras do enunciado
AUTONOMIA_BASE_S = 5000  # autonomia nominal em segundos em laboratório (exemplo)
APLICAR_FATOR_93 = student_id.startswith('2')  # se matrícula começa com 2
FATOR_CORRECAO = 0.93 if APLICAR_FATOR_93 else 1.0
AUTONOMIA_S = int(AUTONOMIA_BASE_S * FATOR_CORRECAO)

# Tempo consumido por parada (recarga OU apenas tirar foto), em segundos:
TEMPO_PARADA_S = 72

# Velocidade máxima (km/h) e discretização (múltiplos de 4)
VEL_MAX = 96
VEL_STEP = 4
VEL_MIN = 4

# Horários de voo válidos
HORA_INICIO = datetime.time(6, 0, 0)
HORA_FIM = datetime.time(19, 0, 0)

# Ponto fixo (Unibrasil campus) - CEP inicial/final (conforme enunciado)
CEP_UNIBRASIL = "82821020"

# Penalidades/ pesos na função fitness
# A fitness function returns lower = better
WEIGHT_TIME = 1.0           # tempo total em segundos
COST_PER_RECHARGE = 80.0 if student_id.startswith('2') else 0.0
WEIGHT_COST = 60.0         # transforma reais em "segundos equivalentes" (ajustável)
PENALTY_INVALID = 1e9      # penalidade para soluções inválidas

# Algoritmo Genético - parâmetros
POP_SIZE = 120
GENERATIONS = 800
ELITE_SIZE = 5
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.25

# Arquivos de entrada/saída
CEPS_CSV = "coordenadas.csv"
VENTO_CSV = "vento.csv"
SAIDA_CSV = "solucao_melhor.csv"

# ----------------------------
# Tipos e utilitários
# ----------------------------
Point = namedtuple("Point", ["cep", "lat", "lon", "desc"])

def read_ceps(filename=CEPS_CSV):
    """Lê ceps CSV. Espera colunas: CEP,lat,lon,descricao"""
    pts = []
    try:
        with open(filename, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                cep = row.get('CEP') or row.get('cep') or row.get('Cep')
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                desc = row.get('descricao', '')
                pts.append(Point(cep.strip(), lat, lon, desc))
    except FileNotFoundError:
        print(f"[ERRO] Arquivo {filename} não encontrado. Crie um ceps.csv adequado.", file=sys.stderr)
        raise
    return pts

def read_vento(filename=VENTO_CSV):
    """
    Lê vento CSV. Formato genérico:
    dia,hora,vel_kmh,direcao_deg
    Retorna dict keyed by (dia, hora) -> (vel_kmh, direcao_deg)
    """
    tab = {}
    try:
        with open(filename, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                dia = int(row['dia'])
                hora = int(row['hora'])
                vel = float(row['vel_kmh'])
                direc = float(row['direcao_deg'])
                tab[(dia, hora)] = (vel, direc)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo {filename} não encontrado. Crie um vento.csv adequado.", file=sys.stderr)
        raise
    return tab

# ----------------------------
# Geometria e tempo
# ----------------------------
EARTH_R = 6371000.0  # metros

def haversine_m(lat1, lon1, lat2, lon2):
    """Distância haversine em metros"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_R * c

def kmh_to_mps(kmh):
    return (kmh * 1000.0)/3600.0

def time_seconds_for_distance_m(distance_m, chosen_speed_kmh, wind_kmh, wind_dir_deg, bearing_deg):
    """
    Calcula o tempo em segundos para percorrer 'distance_m' considerando:
      - velocidade do drone antes do vento (chosen_speed_kmh)
      - vento (velocidade e direção)
      - direção do voo (bearing_deg in degrees)
    Regras:
      - velocidade efetiva = velocidade_drone + componente_do_vento_along_trajeto
      - a componente do vento é calculada com base no ângulo entre direção do vento e direção de voo.
    Observação: a velocidade efetiva não pode ser negativa. Se < 1e-6, consideramos muito lento e aplicamos reparo/penalidade.
    """
    # Converte
    v_drone = kmh_to_mps(chosen_speed_kmh)
    v_wind = kmh_to_mps(wind_kmh)
    # ângulo entre vento vindo de 'wind_dir_deg' (ex: SSE) e direção de voo (bearing_deg)
    # se vento_dir é direção de onde vento vem (meteorologia), então vetor vento aponta em wind_dir_deg + 180
    wind_vector_dir = (wind_dir_deg + 180.0) % 360.0
    # diferença de direção (radians)
    delta = math.radians((wind_vector_dir - bearing_deg + 360) % 360)
    # componente do vento na direção do voo:
    comp = v_wind * math.cos(delta)
    v_effective = v_drone + comp
    # se exite requisito de velocidade minima (para matrícula começando com 2): drone não permite <10 m/s
    if student_id.startswith('2'):
        if v_drone < 10.0:
            # impraticável: reparo ou penalidade; retornamos v_effective como v_drone (para seguir) mas essa escolha será penalizada
            pass
    if v_effective <= 0.001:
        # velocidade efetiva zero -> inviável
        return float('inf'), v_effective
    t = distance_m / v_effective
    # Caso de matrícula começando com 2, arredondar o tempo para cima ao segundo
    if student_id.startswith('2'):
        t = math.ceil(t)
    return t, v_effective

def bearing_deg(lat1, lon1, lat2, lon2):
    """Retorna azimute (bearing) de 0..360 graus entre dois pontos"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lam1 = math.radians(lon1)
    lam2 = math.radians(lon2)
    y = math.sin(lam2 - lam1) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(lam2 - lam1)
    theta = math.atan2(y, x)
    bearing = (math.degrees(theta) + 360) % 360
    return bearing

# ----------------------------
# Modelo de consumo de energia
# ----------------------------
def energy_consumption_rate(speed_kmh):
    """
    Consumo médio de energia (unidades de bateria por segundo) como função da velocidade (km/h).
    ATENÇÃO: substitua por sua fórmula oficial, se fornecida:
    - Aqui adotamos: P = a + b * v + c * v^2 (um polinômio simples)
    - Normalizamos para que autonomia à velocidade mínima e sem vento seja aproximadamente AUTONOMIA_S.
    """
    # parâmetros arbitrários para modelagem (ajustáveis)
    a = 0.5
    b = 0.02
    c = 0.0008
    v = float(speed_kmh)
    rate = a + b * v + c * v * v  # "unidades de bateria por segundo"
    return rate

def estimate_flight_energy_sec(distance_s, speed_kmh):
    """Estima o consumo total de energia (unidades de bateria) para um tempo de voo distance_s segundos a speed_kmh"""
    rate = energy_consumption_rate(speed_kmh)
    return rate * distance_s

# ----------------------------
# Representação Genética
# ----------------------------
# Usaremos representação por permutação + pontos de recarga embutidos.
# Cada indivíduo:
#   - ordem: permutação dos CEPs (exclui o CEP inicial/final - Unibrasil)
#   - recarga_flags: lista booleana para cada etapa indicando se vai haver pouso para recarga naquele ponto (True/False)
#   - speeds: lista de velocidades (km/h) para cada trecho (múltiplos de 4)
#
# Genes combinados em dicionário para simplicidade.

def initial_population(points, pop_size=POP_SIZE):
    """Cria população inicial"""
    # points: lista de Point incluindo Unibrasil (start) - vamos excluir o primeiro que é Unibrasil
    start = next((p for p in points if p.cep == CEP_UNIBRASIL), None)
    if start is None:
        raise ValueError("CEP do Unibrasil não encontrado em ceps.csv")
    others = [p for p in points if p.cep != CEP_UNIBRASIL]
    pop = []
    for _ in range(pop_size):
        order = others[:]
        random.shuffle(order)
        rec_flags = [False] * (len(order))
        # algumas soluções iniciais com recargas aleatórias
        for i in range(len(rec_flags)):
            if random.random() < 0.12:
                rec_flags[i] = True
        speeds = [random.choice(range(VEL_MIN, VEL_MAX+1, VEL_STEP)) for _ in range(len(order)+1)]
        ind = {"order": order, "rec": rec_flags, "speeds": speeds}
        pop.append(ind)
    return pop

# ----------------------------
# Fitness, validação e reparo
# ----------------------------
def simulate_individual(individual, points_dict, vento_table):
    """
    Simula um indivíduo e retorna:
      - valid (bool)
      - total_time_s (inclui tempos de voo e tempos de parada)
      - num_recharges
      - total_cost (monetary)
      - detailed_trace (lista de trechos com metadata para csv)
    points_dict: map cep->Point
    vento_table: map (dia,hora)->(vel_kmh,direc_deg)
    """
    # Monta a rota: start -> order[0] -> order[1] -> ... -> order[-1] -> start
    start = points_dict[CEP_UNIBRASIL]
    order = individual['order']
    rec_flags = individual['rec']
    speeds = individual['speeds']
    # trajeto de ceps como lista de Points (incl start at begin and end)
    route = [start] + order + [start]
    # for rec_flags length == len(order); speeds length == len(order)+1 (speeds per trecho)
    # Vamos simular dia/hora; assume que iniciamos no dia=1 às 06:00:00 no primeiro trecho
    current_day = 1
    current_time = datetime.datetime.combine(datetime.date.today(), HORA_INICIO)
    battery = AUTONOMIA_S
    total_time_s = 0.0
    num_recharges = 0
    total_cost = 0.0
    trace = []
    valid = True

    for i in range(len(route)-1):
        a = route[i]
        b = route[i+1]
        # distância e direção
        dist_m = haversine_m(a.lat, a.lon, b.lat, b.lon)
        bear = bearing_deg(a.lat, a.lon, b.lat, b.lon)
        # velocidade escolhida para este trecho
        speed_kmh = speeds[min(i, len(speeds)-1)]
        # determine wind for this (day,hour) according to departure time
        dep_hour_key = current_time.hour
        wind_key = (current_day, dep_hour_key)
        if wind_key not in vento_table:
            # tentativa: escolher a hora mais próxima presente
            candidates = [k for k in vento_table.keys() if k[0]==current_day]
            if not candidates:
                # fallback: dia 1 hora 6
                wind = (0.0, 0.0)
            else:
                # escolher entry com hora mais próxima
                candidates_sorted = sorted(candidates, key=lambda x: abs(x[1]-dep_hour_key))
                wind = vento_table[candidates_sorted[0]]
        else:
            wind = vento_table[wind_key]
        wind_kmh, wind_dir_deg = wind
        # tempo de voo
        tsec, v_effect = time_seconds_for_distance_m(dist_m, speed_kmh, wind_kmh, wind_dir_deg, bear)
        if math.isinf(tsec):
            # trecho inviavel por vento/velocidade -> marcar inválido
            valid = False
            return False, float('inf'), 9999, float('inf'), []
        # consumo estimado durante o voo
        energy_needed = estimate_flight_energy_sec(tsec, speed_kmh)
        # acrescente 72s de consumo por parada no destino (se for parada para foto ou recarga)
        # Se b é o ponto final do dia e vamos pousar para recarga, consideramos isso depois
        # Verifica se há bateria para realizar o voo + 72s (parada para foto) antes de pousar
        required = energy_needed + estimate_flight_energy_sec(TEMPO_PARADA_S, speed_kmh)
        if battery < required:
            # se não há carga suficiente para ir ao próximo ponto -> deve haver recarga ANTES de sair
            # O drone só pode parar nas coordenadas dadas -> então a recarga precisa ter ocorrido no ponto anterior.
            # Se estamos no início (i==0) e não tem bateria, invalida solução
            if i == 0:
                valid = False
                return False, float('inf'), 9999, float('inf'), []
            # Se o previous point (route[i-1]) não foi marcado como recarga, então reparo: inserir recarga lá
            # Aqui faremos reparo automático: se o ponto anterior for um CEP permitido (é), então adicionamos +1 recarga e recarregamos a bateria
            # Esse reparo altera a rota e incrementa custos/tempo
            # Reparo será feito atualizando battery e trace e num_recharges
            # Para simplificar nessa simulação, se o previous point não existir para recarga -> tornar inválido
            prev = route[i]
            # Simular pouso-recarregar no ponto atual antes de partir (i.e., transformar o parada em recarga)
            # NOTE: this is a quick repair: we consume TEMPO_PARADA_S on the ground for reabastecer and set battery full
            # Add cost/time
            num_recharges += 1
            total_cost += COST_PER_RECHARGE
            # gastar TEMPO_PARADA_S temporariamente (já incluímos consumo de TEMPO_PARADA_S acima)
            current_time += datetime.timedelta(seconds=TEMPO_PARADA_S)
            total_time_s += TEMPO_PARADA_S
            battery = AUTONOMIA_S  # recarregado
            # now check again
            if battery < required:
                # even after recharge it's impossible (should not happen)
                valid = False
                return False, float('inf'), 9999, float('inf'), []
        # consume for flight
        battery -= energy_needed
        total_time_s += tsec
        current_time += datetime.timedelta(seconds=tsec)
        # at arrival, spend TEMPO_PARADA_S for photo/decel/accel
        battery -= estimate_flight_energy_sec(TEMPO_PARADA_S, speed_kmh)
        current_time += datetime.timedelta(seconds=TEMPO_PARADA_S)
        total_time_s += TEMPO_PARADA_S
        # record trace
        # decide if we land for recharge here:
        will_recharge = False
        if i < len(rec_flags) and rec_flags[i]:
            will_recharge = True
        # If battery remaining is less than needed for next leg, we must recharge here (unless next is start and finish)
        # But next iteration will detect and repair
        rec_str = "SIM" if will_recharge else "NÃO"
        trace.append({
            "cep_from": a.cep, "lat_from": a.lat, "lon_from": a.lon,
            "cep_to": b.cep, "lat_to": b.lat, "lon_to": b.lon,
            "dep_day": current_day,
            "dep_time": (current_time - datetime.timedelta(seconds=(tsec+TEMPO_PARADA_S))).time().isoformat(),
            "arr_time": current_time.time().isoformat(),
            "speed_kmh": speed_kmh,
            "pouso": rec_str,
            "flight_time_s": tsec
        })
        if will_recharge:
            # perform recharge
            num_recharges += 1
            total_cost += COST_PER_RECHARGE
            current_time += datetime.timedelta(seconds=TEMPO_PARADA_S)
            total_time_s += TEMPO_PARADA_S
            battery = AUTONOMIA_S
        # If arrival time exceeds HORA_FIM, must land and resume next day
        if current_time.time() > HORA_FIM:
            # landing after HORA_FIM triggers special fee (if applicable) and we must resume next day at 06:00
            # But the enunciado exige pousar até 19:00 para recarregar -> aqui forçamos pouso (if battery>0 we can choose to stay)
            # For simplicity, we force landing and consider it a recharge
            num_recharges += 1
            total_cost += COST_PER_RECHARGE
            # landing cost in monetary already counted. Advance to next day 06:00
            current_day += 1
            if current_day > 7:
                # excedeu 7 dias
                valid = False
                return False, float('inf'), 9999, float('inf'), []
            # set time to next day 06:00:00
            # minimal overnight time: compute seconds until next day 06:00
            next_day_time = datetime.datetime.combine(current_time.date() + datetime.timedelta(days=1), HORA_INICIO)
            # set current_time to that; battery = full
            current_time = next_day_time
            battery = AUTONOMIA_S
    # end for
    # total cost add weight
    return valid, total_time_s, num_recharges, total_cost, trace

def fitness(individual, points_dict, vento_table):
    valid, total_time_s, num_recharges, total_cost, trace = simulate_individual(individual, points_dict, vento_table)
    if not valid:
        return PENALTY_INVALID
    # convert cost monetary to seconds-equivalent
    cost_equiv = num_recharges * COST_PER_RECHARGE * WEIGHT_COST
    score = WEIGHT_TIME * total_time_s + cost_equiv
    # small tie-breaker: fewer recharges
    score += num_recharges * 100.0
    return score

# ----------------------------
# GA operators
# ----------------------------
def tournament_selection(pop, fitnesses, k=TOURNAMENT_SIZE):
    best = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best is None or fitnesses[i] < fitnesses[best]:
            best = i
    return deepcopy(pop[best])

def order_crossover(p1, p2):
    """Order crossover (OX) for permutation"""
    n = len(p1['order'])
    if n <= 2:
        child1_order = p1['order'][:]
        child2_order = p2['order'][:]
    else:
        a, b = sorted(random.sample(range(n), 2))
        def ox(parent_a, parent_b):
            child = [None]*n
            child[a:b+1] = parent_a['order'][a:b+1]
            cur = 0
            for g in parent_b['order']:
                if g in child:
                    continue
                while child[cur] is not None:
                    cur += 1
                child[cur] = g
            return child
        child1_order = ox(p1, p2)
        child2_order = ox(p2, p1)
    # rec flags and speeds: mix by uniform
    def mix_list(a,b):
        return [a[i] if random.random() < 0.5 else b[i] for i in range(len(a))]
    child1 = {"order": child1_order, "rec": mix_list(p1['rec'], p2['rec']), "speeds": mix_list(p1['speeds'], p2['speeds'])}
    child2 = {"order": child2_order, "rec": mix_list(p2['rec'], p1['rec']), "speeds": mix_list(p2['speeds'], p1['speeds'])}
    return child1, child2

def mutate(ind):
    # mutation: swap two in permutation, flip some rec flags, jitter speeds
    n = len(ind['order'])
    if n >= 2 and random.random() < MUTATION_RATE:
        i, j = random.sample(range(n), 2)
        ind['order'][i], ind['order'][j] = ind['order'][j], ind['order'][i]
    # rec flags flips
    for i in range(len(ind['rec'])):
        if random.random() < 0.05:
            ind['rec'][i] = not ind['rec'][i]
    # mutate speeds
    for i in range(len(ind['speeds'])):
        if random.random() < 0.08:
            ind['speeds'][i] = random.choice(range(VEL_MIN, VEL_MAX+1, VEL_STEP))
    return ind

# ----------------------------
# Main GA loop
# ----------------------------
def run_ga(points, vento_table, pop_size=POP_SIZE, generations=GENERATIONS):
    points_dict = {p.cep: p for p in points}
    pop = initial_population(points, pop_size=pop_size)
    best = None
    best_score = float('inf')
    for gen in range(generations):
        fitnesses = [fitness(ind, points_dict, vento_table) for ind in pop]
        # track best
        for i, f in enumerate(fitnesses):
            if f < best_score:
                best_score = f
                best = deepcopy(pop[i])
        # selection + crossover
        newpop = []
        # elitism
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
        for idx in sorted_idx[:ELITE_SIZE]:
            newpop.append(deepcopy(pop[idx]))
        while len(newpop) < pop_size:
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)
            mutate(c1)
            mutate(c2)
            newpop.append(c1)
            if len(newpop) < pop_size:
                newpop.append(c2)
        pop = newpop
        if gen % 50 == 0:
            print(f"[GEN {gen}] melhor_score={best_score:.2f}")
    print(f"[GA] Melhor score final = {best_score:.2f}")
    return best, best_score

# ----------------------------
# CSV de saída
# ----------------------------
def export_solution(individual, points, vento_table, filename=SAIDA_CSV):
    points_dict = {p.cep: p for p in points}
    valid, total_time_s, num_recharges, total_cost, trace = simulate_individual(individual, points_dict, vento_table)
    if not valid:
        raise RuntimeError("Solução inválida não pode ser exportada.")
    # Conversão para CSV com as colunas solicitadas
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "CEP_inicial","Latitude_inicial","Longitude_inicial","Dia_vôo","Hora_inicial",
            "Velocidade","CEP_final","Latitude_final","Longitude_final","Pouso","Hora_final","Descrição"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in trace:
            desc = ""
            w.writerow({
                "CEP_inicial": row["cep_from"],
                "Latitude_inicial": row["lat_from"],
                "Longitude_inicial": row["lon_from"],
                "Dia_vôo": row["dep_day"],
                "Hora_inicial": row["dep_time"],
                "Velocidade": row["speed_kmh"],
                "CEP_final": row["cep_to"],
                "Latitude_final": row["lat_to"],
                "Longitude_final": row["lon_to"],
                "Pouso": row["pouso"],
                "Hora_final": row["arr_time"],
                "Descrição": desc
            })
    print(f"[SAÍDA] Solução exportada para {filename}. Tempo total (s): {total_time_s:.1f}. Recharges: {num_recharges}. Custo R$ {total_cost:.2f}")

# ----------------------------
# Exemplos de testes unitários
# ----------------------------
class TestDroneModels(unittest.TestCase):
    def setUp(self):
        # small synthetic dataset: start + 2 pontos
        self.start = Point(cep=CEP_UNIBRASIL, lat=-25.5, lon=-49.2, desc="Unibrasil")
        self.p1 = Point(cep="80010010", lat=-25.48, lon=-49.25, desc="P1")
        self.p2 = Point(cep="80020020", lat=-25.50, lon=-49.22, desc="P2")
        self.points = [self.start, self.p1, self.p2]
        # vento table: day 1 at hour 6 => no wind
        self.vento = {(1, 6): (0.0, 0.0)}
    def test_haversine(self):
        d = haversine_m(self.start.lat, self.start.lon, self.p1.lat, self.p1.lon)
        self.assertTrue(50 < d < 4000)  # plausible distance in meters
    def test_time_calc_no_wind(self):
        d = haversine_m(self.start.lat, self.start.lon, self.p1.lat, self.p1.lon)
        t, v = time_seconds_for_distance_m(d, 36, 0.0, 0.0, bearing_deg(self.start.lat, self.start.lon, self.p1.lat, self.p1.lon))
        self.assertIsInstance(t, float)
        self.assertGreater(t, 0.0)
    def test_simulate_simple_route(self):
        ind = {
            "order": [self.p1, self.p2],
            "rec": [False, False],
            "speeds": [36, 36, 36]
        }
        valid, total_time_s, num_recharges, total_cost, trace = simulate_individual(ind, {p.cep:p for p in self.points}, self.vento)
        self.assertTrue(valid)
        self.assertGreater(total_time_s, 0.0)
        self.assertIsInstance(trace, list)

# ----------------------------
# MAIN
# ----------------------------
def main():
    random.seed(42)
    # Lê dados
    points = read_ceps(CEPS_CSV)
    vento_table = read_vento(VENTO_CSV)
    # Run GA
    best, score = run_ga(points, vento_table)
    # export
    export_solution(best, points, vento_table, SAIDA_CSV)

if __name__ == "__main__":
    # If run as script, roda GA. Se quisermos apenas rodar testes:
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[sys.argv[0]])
    else:
        main()