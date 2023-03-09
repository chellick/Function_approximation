# Function approximation

Задача аппроксимации функции заключается в ...  

Коэффициенты для сплайн-интерполяции находятся путем решения системы линейных уравнений, составленной из условий, которые гарантируют гладкость и монотонность сплайнов.

Пусть у нас есть набор данных (x_0, y_0), (x_1, y_1), ..., (x_n, y_n), где x_i - узлы интерполяции, y_i - соответствующие значения функции, которую нужно интерполировать.

Для каждого интервала [x_i, x_{i+1}] мы находим уравнение кубического сплайна, которое имеет следующий вид:

$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$


где $S_i(x)$ - кубический сплайн на интервале $[x_i, x_{i+1}], a_i, b_i, c_i, d_i$ - коэффициенты, которые нужно найти.

Задача состоит в том, чтобы найти значения коэффициентов a_i, b_i, c_i, d_i для каждого интервала $[x_i, x_{i+1}]$, такие, чтобы выполнялись следующие условия:

Сплайны должны проходить через узлы интерполяции: $S_i(x_i) = y_i$.
Сплайны должны быть гладкими: $S_{i-1}'(x_i) = S_i'(x_i)$.
Сплайны должны быть дважды гладкими: $S_{i-1}''(x_i) = S_i''(x_i)$.
Для сплайнов, которые используются для интерполяции монотонной функции, сплайны должны быть монотонными: если $y_{i+1} > y_i, то S_i'(x_i) >= S_{i-1}'(x_i)$.
После того, как система уравнений составлена, ее можно решить, например, с помощью метода прогонки. Полученные значения коэффициентов $a_i, b_i, c_i, d_i$ могут быть использованы для вычисления значений сплайнов в любой точке на соответствующем интервале.



--------------------------



Сплайн-интерполяция - это метод интерполяции, который состоит в аппроксимации функции кусочно-полиномиальными функциями, называемыми сплайнами. Каждый сплайн определяется на отрезке $[x_i, x_{i+1}]$ и имеет степень $k$, где $k$ - натуральное число.




Алгоритм сплайн-интерполяции состоит из нескольких этапов:

Подготовка данных: имея набор точек $(x_i, y_i)$, необходимо отсортировать их по возрастанию $x_i$, затем рассчитать разности $\Delta x_i = x_{i+1} - x_i$ и $\Delta y_i = y_{i+1} - y_i$.

Определение коэффициентов: необходимо определить коэффициенты $a_i, b_i, c_i, d_i$ для каждого сплайна на отрезке $[x_i, x_{i+1}]$. Для этого необходимо решить систему уравнений, составленную из условий:

$\left[
    \begin{gathered}
    S_i(x_i) = y_i \\
    S_i(x_{i+1}) = y_{i+1} \\
    S_i'(x_{i+1}) = S_{i+1}'(x_{i+1}) \\
    S_i''(x_{i+1}) = S_{i+1}''(x_{i+1})\\
    \end{gathered}
\right. $

Здесь $S_i(x)$ - сплайн на отрезке $[x_i, x_{i+1}]$, а $S_i'(x)$ и $S_i''(x)$
- его первая и вторая производные соответственно.

Решив эту систему уравнений, мы получим следующие значения для коэффициентов сплайна:

$a_i = y_i$

$b_i = \frac{\Delta y_i}{\Delta x_i} - \frac{\Delta x_i}{3}(c_{i+1} + 2c_i)$

$d_i = \frac{c_{i+1} - c_i}{3\Delta x_i}$

$c_i$ определяется из рекуррентной формулы:

$$c_i = \frac{\Delta y_i/\Delta x_i - \Delta y_{i-1}/\Delta x_{i-1}}{\Delta x_i/\Delta x_{i-1} + 2}$$

Здесь $c_0$ и $c_n$ вычисляются с помощью естественных граничных условий:

$c_0 = 0$
$c_n = 0$
Коэффициенты $a_i$ соответствуют значению функции на левой границе отрезка сплайна, $b_i$ - линейной части функции на этом отрезке, $c_i$ - квадратичной части функции на отрезке, а $d_i$ - кубической части функции на этом отрезке.

Построение сплайнов: используя полученные коэффициенты, мы можем построить кусочно-полиномиальные функции, соответствующие каждому сплайну на отрезке $[x_i, x_{i+1}]$. Эти функции могут быть заданы следующим образом:
$$S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3$$

Теперь, чтобы посчитать значение функции в произвольной точке $x$, необходимо определить, на каком отрезке $[x_i, x_{i+1}]$ она находится, а затем вычислить значение соответствующего сплайна $S_i(x)$ в этой точке.

Важно отметить, что сплайны обычно используются для интерполяции функций, но могут также использоваться для аппроксимации. При интерполяции мы строим сплайны, проходящие через все заданные точки, в то время как при аппроксимации мы строим сплайны, которые лучше всего приближают исходную функцию, но не обязательно проходят через все точки.
