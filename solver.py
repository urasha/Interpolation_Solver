import csv
import math
from functools import reduce

FUNCTION_MAP = {
    "sin(x)": math.sin,
    "cos(x)": math.cos,
    "exp(x)": math.exp,
}


def compute_diff_table(points):
    n = len(points)
    table = [[y for _, y in points]]
    for level in range(1, n):
        prev = table[-1]
        curr = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
        table.append(curr)
    return table


def interp_lagrange(points, x0):
    result = 0.0
    for i, (xi, yi) in enumerate(points):
        term = yi
        for j, (xj, _) in enumerate(points):
            if i != j:
                term *= (x0 - xj) / (xi - xj)
        result += term
    return result


def interp_newton(points, x0):
    n = len(points)
    xs = [p[0] for p in points]

    dd = [[p[1]] for p in points]
    for level in range(1, n):
        for i in range(n - level):
            num = dd[i + 1][level - 1] - dd[i][level - 1]
            den = xs[i + level] - xs[i]
            dd[i].append(num / den)

    result = dd[0][0]
    prod = 1.0
    for level in range(1, n):
        prod *= (x0 - xs[level - 1])
        result += dd[0][level] * prod
    return result


def interp_gauss(points, x0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs) - 1
    mid = n // 2

    fin_diffs = [ys[:]]
    for k in range(1, n + 1):
        prev = fin_diffs[-1]
        fin_diffs.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])

    h = xs[1] - xs[0]

    shifts = [0, -1, 1, -2, 2, -3, 3, -4, 4][:n + 1]

    def branch_positive(x):
        t = (x - xs[mid]) / h
        terms = []
        for k in range(1, n + 1):
            prod = reduce(lambda a, b: a * b,
                          [(t + shifts[j]) for j in range(k)], 1.0)
            delta = fin_diffs[k][len(fin_diffs[k]) // 2]
            terms.append(prod * delta / math.factorial(k))
        return ys[mid] + sum(terms)

    def branch_negative(x):
        t = (x - xs[mid]) / h
        terms = []
        for k in range(1, n + 1):
            prod = reduce(lambda a, b: a * b,
                          [(t - shifts[j]) for j in range(k)], 1.0)
            offset = (1 - len(fin_diffs[k]) % 2)
            delta = fin_diffs[k][len(fin_diffs[k]) // 2 - offset]
            terms.append(prod * delta / math.factorial(k))
        return ys[mid] + sum(terms)

    return branch_positive(x0) if x0 > xs[mid] else branch_negative(x0)


def interp_stirling(points, x0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    h = xs[1] - xs[0]
    diff = compute_diff_table(points)

    n = len(xs) - 1
    alpha = n // 2

    t = (x0 - xs[alpha]) / h

    shifts = [0]
    for i in range(1, n + 1):
        shifts += [-i, i]
    shifts = shifts[:n]

    s_pos = ys[alpha]
    s_neg = ys[alpha]
    prod_pos = 1.0
    prod_neg = 1.0
    fact = 1.0

    for k in range(1, n + 1):
        fact *= k
        shift = shifts[k - 1]

        prod_pos *= (t + shift)
        prod_neg *= (t - shift)

        col = diff[k]
        idx_center = len(col) // 2
        delta_center = col[idx_center]

        offset = 1 - (len(col) % 2)
        delta_side = col[idx_center - offset]

        s_pos += prod_pos * delta_center / fact
        s_neg += prod_neg * delta_side / fact

    return 0.5 * (s_pos + s_neg)


def interp_bessel(points, x0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    h = xs[1] - xs[0]
    diff = compute_diff_table(points)

    n = len(xs)
    m = n // 2 - 1

    t = (x0 - xs[m]) / h

    result = 0.5 * (ys[m] + ys[m + 1])
    result += (t - 0.5) * diff[1][m]

    even_coeff = t * (t - 1) / 2
    odd_coeff = (t - 0.5) * t * (t - 1) / 6

    r = 1
    while True:
        k_even = 2 * r
        k_odd = k_even + 1

        if k_even < len(diff):
            left = m - r
            right = left + 1
            if 0 <= left and right < len(diff[k_even]):
                avg = 0.5 * (diff[k_even][left] + diff[k_even][right])
                result += even_coeff * avg

        if k_odd < len(diff):
            idx = m - r
            if 0 <= idx < len(diff[k_odd]):
                result += odd_coeff * diff[k_odd][idx]

        if k_even >= len(diff) and k_odd >= len(diff):
            break

        if m - r - 1 < 0:
            break

        even_coeff *= (t + r) * (t - r - 1) / ((2 * r + 2) * (2 * r + 1))
        odd_coeff *= (t + r) * (t - r - 1) / ((2 * r + 3) * (2 * r + 2))

        r += 1

    return result


def process_data(kind, data, methods, x_star, gui):
    try:
        if kind == 'file':
            points = []
            with open(data, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        x_val = float(row[0])
                        y_val = float(row[1])
                        points.append((x_val, y_val))

        elif kind == 'func':
            name = data['name']
            left = data['left']
            right = data['right']
            count = data['n']
            func = FUNCTION_MAP[name]
            step = (right - left) / (count - 1)
            points = [(left + i * step, func(left + i * step))
                      for i in range(count)]
        else:
            points = list(data)

        points.sort(key=lambda p: p[0])

    except Exception as e:
        gui.show_error(f"Ошибка подготовки данных: {e}")
        return

    gui.clear_diff_table()
    gui.clear_results()

    diffs = compute_diff_table(points)
    gui.update_diff_table(diffs)

    try:
        if methods.get('lagrange'):
            y = interp_lagrange(points, x_star)
            gui.add_result('Лагранж', f"{y:.6f}")
        if methods.get('newton'):
            y = interp_newton(points, x_star)
            gui.add_result('Ньютон', f"{y:.6f}")
        if methods.get('gauss'):
            y = interp_gauss(points, x_star)
            gui.add_result('Гаусс', f"{y:.6f}")
        if methods.get('stirling'):
            y = interp_stirling(points, x_star)
            gui.add_result('Стирлинг', f"{y:.6f}")
        if methods.get('bessel'):
            y = interp_bessel(points, x_star)
            gui.add_result('Бессель', f"{y:.6f}")
    except Exception as e:
        gui.show_error(f"Ошибка вычислений: {e}")
        return

    try:
        gui.plot(points, x_star)
    except AttributeError:
        pass

    gui.show_ok("Готово")
