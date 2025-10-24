import pandas as pd

ceps = pd.read_csv('ceps.csv')
start_cep = '82821020'
start_index = ceps[ceps['CEP'] == start_cep].index[0]
num_points = len(ceps)

wind_data = {
    1: {
        6: {'speed': 17, 'dir': 'E-NE'},
        9: {'speed': 18, 'dir': 'E'},
        12: {'speed': 19, 'dir': 'E'},
        15: {'speed': 19, 'dir': 'E'},
        18: {'speed': 20, 'dir': 'E'},
        21: {'speed': 20, 'dir': 'E'}
    },
    # Adicione os outros dias de forma similar, usando os valores das tabelas (Dia 2: 20,19,16,19,21,21 com dirs 'E'; e assim por diante)
    7: {
        6: {'speed': 6, 'dir': 'NE'},
        9: {'speed': 8, 'dir': 'NE'},
        12: {'speed': 14, 'dir': 'NE'},
        15: {'speed': 16, 'dir': 'E-NE'},
        18: {'speed': 13, 'dir': 'E-NE'},
        21: {'speed': 10, 'dir': 'E-NE'}
    }
}

dir_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'E-NE': 67.5, 'E': 90, 'ESE': 112.5,
    'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'W-SW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

import math
from datetime import datetime, timedelta

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.atan2(y, x)
    brng = math.degrees(brng)
    return (brng + 360) % 360

def get_wind(day, dep_time):
    h = dep_time.hour
    columns = [6, 9, 12, 15, 18, 21]
    col = max([c for c in columns if c <= h] or [6])
    return wind_data[day][col]

def effective_speed(v_d, bearing_deg, wind_speed, wind_dir):
    theta = bearing_deg
    dir_from = dir_map[wind_dir]
    dir_to = (dir_from + 180) % 360
    v_wx = wind_speed * math.sin(math.radians(dir_to))
    v_wy = wind_speed * math.cos(math.radians(dir_to))
    v_dx = v_d * math.sin(math.radians(theta))
    v_dy = v_d * math.cos(math.radians(theta))
    v_gx = v_dx + v_wx
    v_gy = v_dy + v_wy
    return math.sqrt(v_gx**2 + v_gy**2)

BASE_AUTONOMY = 5000  # segundos
REF_SPEED = 36  # km/h
STOP_CONSUME = 72  # unidades
# Se matrícula começa com 2, aplique BASE_AUTONOMY *= 0.93

def calculate_segment(from_idx, to_idx, v_d, day, dep_time):
    lat1, lon1 = ceps.iloc[from_idx]['Latitude'], ceps.iloc[from_idx]['Longitude']
    lat2, lon2 = ceps.iloc[to_idx]['Latitude'], ceps.iloc[to_idx]['Longitude']
    dist = haversine(lat1, lon1, lat2, lon2)
    brng = bearing(lat1, lon1, lat2, lon2)
    wind = get_wind(day, dep_time)
    v_eff = effective_speed(v_d, brng, wind['speed'], wind['dir'])
    time_flight_sec = math.ceil((dist / v_eff) * 3600)  # arredonde para cima se matrícula 2
    consume_rate = (v_d / REF_SPEED) ** 2
    consume_flight = time_flight_sec * consume_rate
    consume_total = consume_flight + STOP_CONSUME
    arr_time = dep_time + timedelta(seconds=time_flight_sec + 72)  # adicione tempo de parada
    return time_flight_sec, consume_total, arr_time, v_eff  # retorne para fitness

def create_individual():
    order = list(range(num_points))
    order.remove(start_index)
    random.shuffle(order)
    vels = [random.randint(1, 24) * 4 for _ in range(num_points)]  # n segments
    return (order, vels)


def fitness(individual):
    order, vels = individual
    path = [start_index] + order + [start_index]
    total_time = 0
    total_recharges = 0
    day = 1
    current_time = datetime(2000,1,1,6,0,0).time()  # start 06:00
    battery = BASE_AUTONOMY
    for i in range(len(path) - 1):
        v_d = vels[i]
        time_f, consume, arr_time, _ = calculate_segment(path[i], path[i+1], v_d, day, datetime.combine(datetime.today(), current_time))
        if consume > battery or arr_time.hour >= 19 and arr_time.minute > 0:
            # Recharge: add cost, advance to next day
            total_recharges += 1
            day += 1
            if day > 7: return float('inf')  # invalid
            current_time = datetime(2000,1,1,6,0,0).time()
            battery = BASE_AUTONOMY
            # Recalculate with new day/time if needed, but for simplicity, recharge and retry the segment
            time_f, consume, arr_time, _ = calculate_segment(path[i], path[i+1], v_d, day, datetime.combine(datetime.today(), current_time))
        battery -= consume
        total_time += time_f
        current_time = arr_time.time()
    return total_time + total_recharges * 10000  # peso arbitrário para minimizar recharges

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(range(len(population)), k)
    best = min(selected, key=lambda idx: fitnesses[idx])
    return population[best]

# Implemente PMX crossover para ordem, uniform para vels
# Mutação: com prob 0.05, troque dois em order ou altere vel

def evolve(population, generations=100, pop_size=50):
    for _ in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        new_pop = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)  # implemente
            mutate(child1)
            mutate(child2)
            new_pop.extend([child1, child2])
        population = new_pop
    best = min(population, key=fitness)
    return best

##############################
# TESTES
##############################

# - Deixar comentado por enqunanto

# def test_haversine():
#     dist = haversine(-25.615, -49.372, -25.353, -49.188)
#     assert abs(dist - expected) < 0.1  # use valor esperado

# # Adicione mais

