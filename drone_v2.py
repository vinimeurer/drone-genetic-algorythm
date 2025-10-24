import math
import random
import pandas as pd
from datetime import datetime, timedelta
import copy

# Constants
START_CEP = '82821020'
BASE_AUTONOMY = 5000  # seconds (1h 23m 20s)
AUTONOMY_CORRECTION = 0.93  # Apply if matricula starts with 2, else 1.0
REF_SPEED = 36  # km/h for normal mode
STOP_COST = 72  # seconds per stop (photo or recharge)
MAX_SPEED = 96  # km/h
MIN_SPEED = 4  # km/h, multiples of 4
SPEED_STEP = 4
MIN_SPEED_MS = 10  # m/s = 36 km/h, apply if matricula 2
RECHARGE_COST = 80  # reais per pouso, if matricula 2
LATE_POUSO_COST = 80  # extra if after 17:00, if matricula 2
MAX_DAYS = 7
START_TIME = '06:00:00'
END_TIME = '19:00:00'
LATE_TIME = '17:00:00'
INVALID_PENALTY = 1e9
DEBUG = True

# Set to True if your matricula starts with 2
MATRICULA_2 = True

if MATRICULA_2:
    BASE_AUTONOMY *= AUTONOMY_CORRECTION
    MIN_SPEED = 36  # km/h

# Wind data from tables
wind_data = {
    1: {
        6: {'speed': 17, 'dir': 'E-NE'},
        9: {'speed': 18, 'dir': 'E'},
        12: {'speed': 19, 'dir': 'E'},
        15: {'speed': 19, 'dir': 'E'},
        18: {'speed': 20, 'dir': 'E'},
        21: {'speed': 20, 'dir': 'E'}
    },
    2: {
        6: {'speed': 20, 'dir': 'E'},
        9: {'speed': 19, 'dir': 'E'},
        12: {'speed': 16, 'dir': 'E'},
        15: {'speed': 19, 'dir': 'E'},
        18: {'speed': 21, 'dir': 'E'},
        21: {'speed': 21, 'dir': 'E'}
    },
    3: {
        6: {'speed': 15, 'dir': 'E-NE'},
        9: {'speed': 17, 'dir': 'NE'},
        12: {'speed': 8, 'dir': 'NE'},
        15: {'speed': 20, 'dir': 'E'},
        18: {'speed': 16, 'dir': 'E'},
        21: {'speed': 15, 'dir': 'E-NE'}
    },
    4: {
        6: {'speed': 8, 'dir': 'E-NE'},
        9: {'speed': 11, 'dir': 'E-NE'},
        12: {'speed': 7, 'dir': 'E-NE'},
        15: {'speed': 6, 'dir': 'E-NE'},
        18: {'speed': 11, 'dir': 'E'},
        21: {'speed': 11, 'dir': 'E'}
    },
    5: {
        6: {'speed': 3, 'dir': 'WSW'},
        9: {'speed': 3, 'dir': 'WSW'},
        12: {'speed': 7, 'dir': 'WSW'},
        15: {'speed': 7, 'dir': 'SSW'},
        18: {'speed': 10, 'dir': 'E'},
        21: {'speed': 11, 'dir': 'ENE'}
    },
    6: {
        6: {'speed': 4, 'dir': 'NE'},
        9: {'speed': 5, 'dir': 'NE'},
        12: {'speed': 4, 'dir': 'NE'},
        15: {'speed': 8, 'dir': 'E'},
        18: {'speed': 15, 'dir': 'E'},
        21: {'speed': 15, 'dir': 'E'}
    },
    7: {
        6: {'speed': 6, 'dir': 'NE'},
        9: {'speed': 8, 'dir': 'NE'},
        12: {'speed': 14, 'dir': 'NE'},
        15: {'speed': 16, 'dir': 'E-NE'},
        18: {'speed': 13, 'dir': 'E-NE'},
        21: {'speed': 10, 'dir': 'E-NE'}
    }
}

# Direction to degrees (from North, clockwise)
dir_map = {
    'N': 0,
    'NNE': 22.5,
    'NE': 45,
    'E-NE': 67.5,
    'E': 90,
    'ESE': 112.5,
    'SE': 135,
    'SSE': 157.5,
    'S': 180,
    'SSW': 202.5,
    'SW': 225,
    'WSW': 247.5,
    'W': 270,
    'WNW': 292.5,
    'NW': 315,
    'NNW': 337.5
}

class DroneSimulator:
    def __init__(self, ceps_df):
        self.ceps = ceps_df
        self.start_index = self.ceps[self.ceps['cep'] == START_CEP].index[0]
        self.num_points = len(self.ceps)
        self.dist_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        dist_matrix = [[0] * self.num_points for _ in range(self.num_points)]
        for i in range(self.num_points):
            for j in range(self.num_points):
                if i != j:
                    dist_matrix[i][j] = self.haversine(
                        self.ceps.iloc[i]['Latitude'], self.ceps.iloc[i]['Longitude'],
                        self.ceps.iloc[j]['Latitude'], self.ceps.iloc[j]['Longitude']
                    )
        return dist_matrix

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def bearing(lat1, lon1, lat2, lon2):
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlon = lon2_rad - lon1_rad
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        brng = math.atan2(y, x)
        brng = math.degrees(brng)
        return (brng + 360) % 360

    def get_wind(self, day, hour):
        columns = [6, 9, 12, 15, 18, 21]
        col = max(c for c in columns if c <= hour) if any(c <= hour for c in columns) else columns[0]
        return wind_data.get(day, wind_data[1])[col]  # Default to day 1 if out of range

    def effective_speed(self, drone_speed, bearing_deg, wind_speed, wind_dir):
        wind_deg = dir_map.get(wind_dir, 0)
        wind_from_deg = wind_deg
        wind_to_deg = (wind_from_deg + 180) % 360  # Wind component towards
        v_wx = wind_speed * math.sin(math.radians(wind_to_deg))
        v_wy = wind_speed * math.cos(math.radians(wind_to_deg))
        v_dx = drone_speed * math.sin(math.radians(bearing_deg))
        v_dy = drone_speed * math.cos(math.radians(bearing_deg))
        v_gx = v_dx + v_wx
        v_gy = v_dy + v_wy
        return math.sqrt(v_gx**2 + v_gy**2)

    def calculate_segment(self, from_idx, to_idx, drone_speed, day, dep_time):
        lat1, lon1 = self.ceps.iloc[from_idx]['Latitude'], self.ceps.iloc[from_idx]['Longitude']
        lat2, lon2 = self.ceps.iloc[to_idx]['Latitude'], self.ceps.iloc[to_idx]['Longitude']
        dist = self.dist_matrix[from_idx][to_idx]
        bearing_deg = self.bearing(lat1, lon1, lat2, lon2)
        hour = dep_time.hour
        wind = self.get_wind(day, hour)
        v_eff = self.effective_speed(drone_speed, bearing_deg, wind['speed'], wind['dir'])
        if v_eff == 0:
            v_eff = 0.001  # Avoid division by zero
        flight_time_sec = (dist / v_eff) * 3600
        if MATRICULA_2:
            flight_time_sec = math.ceil(flight_time_sec)
        # Consumption: higher speeds consume more (scale ~ speed^2)
        if drone_speed > REF_SPEED:
            # factor >1 increases consumption (power ~ v^2)
            speed_factor = (drone_speed / REF_SPEED) ** 2
            consume = flight_time_sec * speed_factor + STOP_COST
        else:
            # at or below REF_SPEED assume baseline consumption proportional to time
            consume = flight_time_sec + STOP_COST
        arr_time = dep_time + timedelta(seconds=flight_time_sec + STOP_COST)
        return flight_time_sec, consume, arr_time, MATRICULA_2 and arr_time > datetime.strptime(LATE_TIME, '%H:%M:%S') or False

class GeneticAlgorithm:
    def __init__(self, simulator, pop_size=3, generations=5, mutation_rate=0.1, elite_size=2):
        self.simulator = simulator
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.num_segments = simulator.num_points - 1  # Segments between points

    def create_individual(self):
        order = list(range(self.simulator.num_points))
        order.remove(self.simulator.start_index)
        random.shuffle(order)
        # biased speeds: prefer speeds <= REF_SPEED to avoid huge consumption
        speeds = []
        low_max = min(REF_SPEED, MAX_SPEED)
        high_min = max(REF_SPEED, MIN_SPEED)
        for _ in range(self.num_segments):
            if random.random() < 0.80 and low_max >= MIN_SPEED:
                # 80% choose a conservative speed (<= REF_SPEED)
                speeds.append(random.randrange(MIN_SPEED, low_max + 1, SPEED_STEP))
            else:
                speeds.append(random.randrange(high_min, MAX_SPEED + 1, SPEED_STEP))
        return {'order': order, 'speeds': speeds}

    def fitness(self, individual):
        order = individual['order']
        speeds = individual.get('speeds', [])
        # safety: pad speeds if somehow shorter than needed
        if len(speeds) < self.num_segments:
            speeds = speeds + [speeds[-1] if speeds else REF_SPEED] * (self.num_segments - len(speeds))

        path = [self.simulator.start_index] + order

        def penal(reason, info=None):
            if DEBUG:
                print("== INVALID INDIVIDUAL ==", reason)
                print("Path (indices):", path)
                print("Speeds:", speeds)
                if info:
                    for k, v in info.items():
                        print(f"{k}: {v}")
            return INVALID_PENALTY, []

        total_time = 0
        total_recharges = 0
        total_cost = 0
        day = 1
        current_time = datetime.strptime(START_TIME, '%H:%M:%S')
        battery = BASE_AUTONOMY
        segments = []
        for i in range(len(path) - 1):
            from_idx = path[i]
            to_idx = path[i + 1]
            speed = speeds[i] if i < len(speeds) else REF_SPEED
            try:
                flight_time, consume, arr_time, is_late = self.simulator.calculate_segment(
                    from_idx, to_idx, speed, day, current_time
                )
            except Exception as e:
                return penal("exception in calculate_segment", {"from": from_idx, "to": to_idx, "speed": speed, "day": day, "time": current_time, "exc": repr(e)})
            pouso = 'NÃO'
            extra_cost = 0
            if consume > battery or arr_time > datetime.strptime(END_TIME, '%H:%M:%S'):
                # Recharge at current, start next day
                pouso = 'SIM'
                total_recharges += 1
                if MATRICULA_2:
                    total_cost += RECHARGE_COST
                    if current_time > datetime.strptime(LATE_TIME, '%H:%M:%S'):
                        total_cost += LATE_POUSO_COST
                day += 1
                if day > MAX_DAYS:
                    return penal("exceeded MAX_DAYS", {"day": day, "MAX_DAYS": MAX_DAYS})
                current_time = datetime.strptime(START_TIME, '%H:%M:%S')
                battery = BASE_AUTONOMY
                # Recalculate segment with new day/time
                try:
                    flight_time, consume, arr_time, is_late = self.simulator.calculate_segment(
                        from_idx, to_idx, speed, day, current_time
                    )
                except Exception as e:
                    return penal("exception in recalc calculate_segment", {"from": from_idx, "to": to_idx, "speed": speed, "day": day, "time": current_time, "exc": repr(e)})
                if consume > battery:  # Still can't, penalize but keep searching
                    return penal("consume > battery even after recharge", {
                        "from": from_idx, "to": to_idx, "speed": speed, "day": day,
                        "flight_time": flight_time, "consume": consume, "battery": battery,
                        "dist": self.simulator.dist_matrix[from_idx][to_idx]
                    })
            battery -= consume
            total_time += flight_time
            # Record segment
            segments.append({
                'cep_inicial': self.simulator.ceps.iloc[from_idx]['CEP'],
                'lat_inicial': self.simulator.ceps.iloc[from_idx]['Latitude'],
                'lon_inicial': self.simulator.ceps.iloc[from_idx]['Longitude'],
                'dia': day,
                'hora_inicial': current_time.strftime('%H:%M:%S'),
                'velocidade': speed,
                'cep_final': self.simulator.ceps.iloc[to_idx]['CEP'],
                'lat_final': self.simulator.ceps.iloc[to_idx]['Latitude'],
                'lon_final': self.simulator.ceps.iloc[to_idx]['Longitude'],
                'pouso': pouso,
                'hora_final': arr_time.strftime('%H:%M:%S')
            })
            current_time = arr_time
            if MATRICULA_2 and is_late:
                extra_cost += LATE_POUSO_COST
            total_cost += extra_cost
        # Return to start
        from_idx = path[-1]
        to_idx = self.simulator.start_index
        speed = random.randrange(MIN_SPEED, MAX_SPEED + 1, SPEED_STEP)  # Random for return
        try:
            flight_time, consume, arr_time, is_late = self.simulator.calculate_segment(
                from_idx, to_idx, speed, day, current_time
            )
        except Exception as e:
            return penal("exception in calculate_segment return", {"from": from_idx, "to": to_idx, "speed": speed, "day": day, "time": current_time, "exc": repr(e)})
        pouso = 'NÃO'
        if consume > battery or arr_time > datetime.strptime(END_TIME, '%H:%M:%S'):
            pouso = 'SIM'
            total_recharges += 1
            if MATRICULA_2:
                total_cost += RECHARGE_COST
            day += 1
            if day > MAX_DAYS:
                return penal("exceeded MAX_DAYS on return", {"day": day})
            current_time = datetime.strptime(START_TIME, '%H:%M:%S')
            battery = BASE_AUTONOMY
            try:
                flight_time, consume, arr_time, is_late = self.simulator.calculate_segment(
                    from_idx, to_idx, speed, day, current_time
                )
            except Exception as e:
                return penal("exception in recalc calculate_segment return", {"from": from_idx, "to": to_idx, "speed": speed, "day": day, "time": current_time, "exc": repr(e)})
            if consume > battery:
                return penal("consume > battery on return even after recharge", {
                    "from": from_idx, "to": to_idx, "speed": speed, "day": day,
                    "flight_time": flight_time, "consume": consume, "battery": battery,
                    "dist": self.simulator.dist_matrix[from_idx][to_idx]
                })
        battery -= consume
        total_time += flight_time
        segments.append({
            'cep_inicial': self.simulator.ceps.iloc[from_idx]['CEP'],
            'lat_inicial': self.simulator.ceps.iloc[from_idx]['Latitude'],
            'lon_inicial': self.simulator.ceps.iloc[from_idx]['Longitude'],
            'dia': day,
            'hora_inicial': current_time.strftime('%H:%M:%S'),
            'velocidade': speed,
            'cep_final': self.simulator.ceps.iloc[to_idx]['CEP'],
            'lat_final': self.simulator.ceps.iloc[to_idx]['Latitude'],
            'lon_final': self.simulator.ceps.iloc[to_idx]['Longitude'],
            'pouso': pouso,
            'hora_final': arr_time.strftime('%H:%M:%S')
        })
        # Fitness: total_time + penalty for recharges + cost
        fitness_value = total_time + total_recharges * 3600  # Arbitrary penalty, 1 hour per recharge
        if MATRICULA_2:
            fitness_value += total_cost * 10  # Arbitrary to include money
        return fitness_value, segments

    def pmx_crossover(self, parent1, parent2):
        # PMX implementation that works with arbitrary value labels (not assuming values == 0..n-1)
        size = len(parent1)
        # prepare child with placeholders
        child = [None] * size
        # choose crossover points
        cxpoint1 = random.randint(0, size - 1)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        # copy slice from parent1 to child
        for i in range(cxpoint1, cxpoint2 + 1):
            child[i] = parent1[i]
        # fill remaining positions using mapping from parent2
        for i in range(cxpoint1, cxpoint2 + 1):
            val = parent2[i]
            if val in child:
                continue
            pos = i
            # find target position by following mapping chain
            while True:
                mapped_val = parent1[pos]
                pos = parent2.index(mapped_val)
                if child[pos] is None:
                    child[pos] = val
                    break
        # fill any remaining None with parent2 values in order
        for i in range(size):
            if child[i] is None:
                for val in parent2:
                    if val not in child:
                        child[i] = val
                        break
        return child

    def crossover(self, parent1, parent2):
        order1 = parent1['order']
        order2 = parent2['order']
        child_order = self.pmx_crossover(order1, order2)
        # For speeds, two-point crossover
        speeds1 = parent1['speeds']
        speeds2 = parent2['speeds']
        size = len(speeds1)
        cx1 = random.randint(0, size - 1)
        cx2 = random.randint(cx1, size - 1)
        child_speeds = speeds1[:cx1] + speeds2[cx1:cx2 + 1] + speeds1[cx2 + 1:]
        return {'order': child_order, 'speeds': child_speeds}

    def mutate(self, individual):
        order = individual['order']
        speeds = individual['speeds']
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
        # mutate speeds with same conservative bias
        low_max = min(REF_SPEED, MAX_SPEED)
        high_min = max(REF_SPEED, MIN_SPEED)
        for k in range(len(speeds)):
            if random.random() < self.mutation_rate:
                if random.random() < 0.80 and low_max >= MIN_SPEED:
                    speeds[k] = random.randrange(MIN_SPEED, low_max + 1, SPEED_STEP)
                else:
                    speeds[k] = random.randrange(high_min, MAX_SPEED + 1, SPEED_STEP)
        return individual

    def evolve(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        for gen in range(self.generations):
            fitnesses = []
            for ind in population:
                try:
                    fit, _ = self.fitness(ind)
                except Exception:
                    fit = INVALID_PENALTY
                    if DEBUG:
                        print("fitness exception for individual")
                fitnesses.append(fit)
            elites = sorted(range(len(fitnesses)), key=lambda x: fitnesses[x])[:self.elite_size]
            new_pop = [copy.deepcopy(population[i]) for i in elites]
            while len(new_pop) < self.pop_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_pop.append(child)
            population = new_pop
            best_fit = min(fitnesses)
            if DEBUG and best_fit >= INVALID_PENALTY:
                print(f"Generation {gen + 1}, all candidates invalid (penalty={INVALID_PENALTY}).")
            print(f"Generation {gen + 1}, Best fitness: {best_fit}")
        # evaluate final population to pick best (guard against all invalid)
        final_scores = []
        for ind in population:
            try:
                final_scores.append(self.fitness(ind)[0])
            except Exception:
                final_scores.append(INVALID_PENALTY)
        best_idx = min(range(len(population)), key=lambda x: final_scores[x])
        if final_scores[best_idx] >= INVALID_PENALTY:
            print("Aviso: todas as soluções têm penalidade (possível configuração impossível).")
        return population[best_idx]

    def tournament_selection(self, population, fitnesses, tour_size=3):
        selected = random.sample(range(len(population)), tour_size)
        best = min(selected, key=lambda x: fitnesses[x])
        return population[best]

# Main
# if __name__ == "__main__":
#     # Load CEPs (user provides ceps.csv)
#     ceps_df = pd.read_csv('coordenadas.csv')  # Columns: CEP, Latitude, Longitude
#     ceps_df['latitude'] = ceps_df['latitude'].astype(float)
#     ceps_df['longitude'] = ceps_df['longitude'].astype(float)

#     simulator = DroneSimulator(ceps_df)
#     ga = GeneticAlgorithm(simulator, pop_size=100, generations=200, mutation_rate=0.05)
#     best_individual = ga.evolve()
#     best_fitness, best_segments = ga.fitness(best_individual)

#     # Output CSV
#     output_df = pd.DataFrame(best_segments)
#     output_df.to_csv('solution.csv', index=False)
#     print("Best solution saved to solution.csv")

if __name__ == "__main__":
    import sys

    # Load CEPs (user provides ceps.csv)
    ceps_df = pd.read_csv('coordenadas.csv')  # Columns: CEP, Latitude, Longitude
    # Normalize column names (strip whitespace)
    ceps_df.columns = [c.strip() for c in ceps_df.columns]

    # Find source column names ignoring case
    col_map = {}
    for col in ceps_df.columns:
        low = col.lower()
        if low == 'cep':
            col_map['cep'] = col
        if low == 'latitude':
            col_map['latitude'] = col
        if low == 'longitude':
            col_map['longitude'] = col

    # Validate required columns
    missing = [k for k in ('cep','latitude','longitude') if k not in col_map]
    if missing:
        print(f"Erro: colunas faltando no coordenadas.csv: {missing}")
        print("Colunas encontradas:", list(ceps_df.columns))
        sys.exit(1)

    # Create canonical columns expected by the rest of the code
    # CEP as string without non-digits, keep both 'cep' (lower) and 'CEP' (upper) for compatibility
    ceps_df['cep'] = ceps_df[col_map['cep']].astype(str).str.replace(r'\D', '', regex=True)
    ceps_df['CEP'] = ceps_df['cep']
    # Latitude/Longitude as floats and also provide capitalized names used in code
    ceps_df['Latitude'] = ceps_df[col_map['latitude']].astype(float)
    ceps_df['Longitude'] = ceps_df[col_map['longitude']].astype(float)

    # Check START_CEP exists
    if not (ceps_df['cep'] == START_CEP).any():
        print(f"Erro: START_CEP '{START_CEP}' não encontrado em coordenadas.csv.")
        print("Exemplos de CEPs no arquivo:", ceps_df['cep'].drop_duplicates().head(10).tolist())
        sys.exit(1)

    simulator = DroneSimulator(ceps_df)
    ga = GeneticAlgorithm(simulator, pop_size=100, generations=200, mutation_rate=0.05)
    best_individual = ga.evolve()
    best_fitness, best_segments = ga.fitness(best_individual)

    # Output CSV
    output_df = pd.DataFrame(best_segments)
    output_df.to_csv('solution.csv', index=False)
    print("Best solution saved to solution.csv")

# # Unit tests (example, expand as needed)
# def test_haversine():
#     dist = DroneSimulator.haversine(-25.6154928550559, -49.372483, -25.3530998572423, -49.1880231206476)
#     assert abs(dist - 30) < 10  # Approximate

# def test_effective_speed():
#     v_eff = DroneSimulator.effective_speed(None, 36, 39.5, 9, 'SSE')
#     assert abs(v_eff - 41) < 1

# def test_get_wind():
#     wind = DroneSimulator.get_wind(None, 1, 8)
#     assert wind['speed'] == 17  # For day 1, hour 8 -> 6h

# # Run tests
# test_haversine()
# test_effective_speed()
# test_get_wind()
# print("Tests passed.")