import std / [math, strutils]
export math

type
  ## A `Dual` contains a primal (the actual value of the variable) and its derivative
  ## By using operator overloading we accumulate the gradient of each operation on
  ## the fly without having to construct a computation graph.
  Dual* = object
    p*: float
    d*: float

proc `**`*(x, y: float): float = pow(x, y)
proc `^`*(x, y: float): float = pow(x, y)
# Overloading the arithmetic rules
proc initDual*(p, d: float): Dual = Dual(p: p, d: d)
template D*(p, d: float): Dual = initDual(p, d)

# We define each operation first as Dual + generic and handle the two different
# cases in one order. Than define a template to define the other order.
proc `+`*[T](x: Dual, y: T): Dual =
  when T is Dual: D(x.p + y.p, x.d + y.d)
  else: D(x.p + y.float, x.d)
template `+`*[T: not Dual](x: T, y: Dual): Dual = y + x

proc `-`*(x: Dual): Dual = D(-x.p, -x.d)

proc `-`*[T](x: Dual, y: T): Dual =
  when T is Dual: D(x.p - y.p, x.d - y.d)
  else: D(x.p - y, x.d)
template `-`*[T: not Dual](x: T, y: Dual): Dual = -y + x

proc `*`*[T](x: Dual, y: T): Dual =
  when T is Dual: D(x.p * y.p, x.p*y.d + x.d*y.p)
  else: D(x.p * y, x.d * y)
template `*`*[T: not Dual](x: T, y: Dual): Dual = y * x

proc `/`*[T](x: Dual, y: T): Dual =
  when T is Dual: D(x.p/y.p, (y.p*x.d-x.p*y.d)/(y.p^2))
  else: D(x.p/y, (y*x.d)/(y^2))
template `/`*[T: not Dual](x: T, y: Dual): Dual = y^(-1) * x

proc `^`*(x: Dual, k: float): Dual = D(x.p^k,     k*x.p^(k-1)*x.d)
proc `^`*(x: Dual, k: int): Dual   = `^`(x, float(k))
proc `**`*[T: SomeNumber](x: Dual, k: T): Dual = `^`(x, float(k))
proc `+=`*[T](v: var Dual, w: T) = v = v + w

# XXX: should this be for SomeNumber?
proc toDual*(x: SomeNumber): Dual = initDual(x.float, 0.0)


proc pretty*(x: Dual, precision: int): string =
  result = "(p: " & formatBiggestFloat(x.p, precision = precision) &
    ", ∂: " & formatBiggestFloat(x.d, precision = precision) & ")"
proc `$`*(x: Dual): string = pretty(x, precision = -1)

# Overloading the control flow operators
proc `==`*(x: Dual, y: Dual): bool = x.p == y.p
proc `<`*(x: Dual,  y: Dual): bool = x.p < y.p
proc `<`*(x: Dual,  y: SomeNumber): bool = x.p < y
proc `<`*(x: SomeNumber, y: Dual): bool = x < y.p
proc `<=`*(x: Dual, y: Dual): bool = x.p <= y.p
proc abs*(x: Dual): Dual =
  result = if x.p >= 0.0: x else: -x

# Derivative of functions with a scalar argument
template gradient*(f: typed, x: SomeNumber): untyped =
  let w = D(x.float, 1.0) # First derivative dx / dx = 1
  f(w).d

# Gradient of functions with a vector argument
proc gradient*(f: proc(x: openArray[Dual]): Dual, x: openArray[SomeNumber]): seq[float] =
  let nx = x.len
  var d = newSeq[float](nx)
  for i in 0 ..< nx:
    var w = newSeq[Dual](nx)
    for j in 0 ..< nx:
      w[j] = if j == i:
               D(x[i].float, 1.0)
             else:
               D(x[i].float, 0.0)
    d[i] = f(w).d
  result = d

## Generate our trusty table of helper procs w/ derivatives
template genCall(fn, xId, deriv: untyped): untyped =
  proc fn*(x: Dual): Dual =
    let xId = x.p
    result = D(fn(xId), x.d * deriv)

import macros
macro defineSupportedFunctions(body: untyped): untyped =
  result = newStmtList()
  for fn in body:
    doAssert fn.kind == nnkInfix and fn[0].strVal == "->"
    let fnName = fn[1].strVal
    let fnId = ident(fnName)
    # generate the code
    let deriv = fn[2]
    let xId = ident"x"
    let ast = getAst(genCall(fnId, xId, deriv))
    result.add ast
  echo result.repr

## NOTE: some of the following functions are not implemented in Nim atm
defineSupportedFunctions:
  sqrt        ->  1.0 / 2.0 / sqrt(x)
  cbrt        ->  1.0 / 3.0 / (cbrt(x)^2.0)
  #abs2        ->  1.0 * 2.0 * x
  #inv         -> -1.0 * abs2(inv(x))
  ln          ->  1.0 / x
  log10       ->  1.0 / x / ln(10.0)
  log2        ->  1.0 / x / ln(2.0)
  #log1p       ->  1.0 / (x + 1.0)
  exp         ->  exp(x)
  #exp2        ->  ln(2.0) * exp2(x)
  #expm1       ->  exp(x)
  sin         ->  cos(x)
  cos         -> -sin(x)
  tan         ->  (1.0 + (tan(x)^2))
  sec         ->  sec(x) * tan(x)
  csc         -> -csc(x) * cot(x)
  cot         -> -(1.0 + (cot(x)^2))
  #sind        ->  Pi / 180.0 * cosd(x)
  #cosd        -> -Pi / 180.0 * sind(x)
  #tand        ->  Pi / 180.0 * (1.0 + (tand(x)^2))
  #secd        ->  Pi / 180.0 * secd(x) * tand(x)
  #cscd        -> -Pi / 180.0 * cscd(x) * cotd(x)
  #cotd        -> -Pi / 180.0 * (1.0 + (cotd(x)^2))
  arcsin      ->  1.0 / sqrt(1.0 - (x^2))
  arccos      -> -1.0 / sqrt(1.0 - (x^2))
  arctan      ->  1.0 / (1.0 + (x^2))
  arcsec      ->  1.0 / abs(x) / sqrt(x^2 - 1.0)
  arccsc      -> -1.0 / abs(x) / sqrt(x^2 - 1.0)
  arccot      -> -1.0 / (1.0 + (x^2))
  #arcsind     ->  180.0 / Pi / sqrt(1.0 - (x^2))
  #arccosd     -> -180.0 / Pi / sqrt(1.0 - (x^2))
  #arctand     ->  180.0 / Pi / (1.0 + (x^2))
  #arcsecd     ->  180.0 / Pi / abs(x) / sqrt(x^2 - 1.0)
  #arccscd     -> -180.0 / Pi / abs(x) / sqrt(x^2 - 1.0)
  #arccotd     -> -180.0 / Pi / (1.0 + (x^2))
  sinh        ->  cosh(x)
  cosh        ->  sinh(x)
  tanh        ->  sech(x)^2
  sech        -> -tanh(x) * sech(x)
  csch        -> -coth(x) * csch(x)
  coth        -> -(csch(x)^2)
  arcsinh     ->  1.0 / sqrt(x^2 + 1.0)
  arccosh     ->  1.0 / sqrt(x^2 - 1.0)
  arctanh     ->  1.0 / (1.0 - (x^2))
  arcsech     -> -1.0 / x / sqrt(1.0 - (x^2))
  arccsch     -> -1.0 / abs(x) / sqrt(1.0 + (x^2))
  arccoth     ->  1.0 / (1.0 - (x^2))
  degToRad    ->  Pi / 180.0
  radToDeg    ->  180.0 / Pi
  erf         ->  2.0 * exp(-x*x) / sqrt(Pi)
  #erfinv      ->  0.5 * sqrt(Pi) * exp(erfinv(x) * erfinv(x))
  erfc        -> -2.0 * exp(-x*x) / sqrt(Pi)
  #erfcinv     -> -0.5 * sqrt(Pi) * exp(erfcinv(x) * erfcinv(x))
  #erfi        ->  2.0 * exp(x*x) / sqrt(Pi)
  #gamma       ->  digamma(x) * gamma(x)
  #lgamma      ->  digamma(x)
  #digamma     ->  trigamma(x)
  #invdigamma  ->  inv(trigamma(invdigamma(x)))
  #trigamma    ->  polygamma(2.0 x)
  #airyai      ->  airyaiprime(x)
  #airybi      ->  airybiprime(x)
  #airyaiprime ->  x * airyai(x)
  #airybiprime ->  x * airybi(x)
  #besselj0    -> -besselj1(x)
  #besselj1    ->  (besselj0(x) - besselj(2.0, x)) / 2.0
  #bessely0    -> -bessely1(x)
  #bessely1    ->  (bessely0(x) - bessely(2.0, x)) / 2.0
  #erfcx       ->  (2.0 * x * erfcx(x) - 2.0 / sqrt(Pi))
  #dawson      ->  (1.0 - 2.0 * x * dawson(x))

when isMainModule:
  # Examples
  echo D(5.0, 1.0) + D(7.0, 0.0) # 12.0 + 1.0ε
  echo D(5.0, 1.0) * 5.0 # 25.0 + 5.0ε
  echo D(5.0, 1.0) * D(7.0, 0.0) # 35.0 + 7.0ε
  echo D(5.0, 1.0)^3 # 125.0 + 75.0ε


  # Function examples
  echo sin(D(1.0, 1.0)) # 0.841 + 0.540ε
  echo exp(ln(D(2.0, 1.5))) # 2.0 + 1.5ε
  # Custom function examples
  proc sq(x: Dual): Dual = x * x
  echo sq(D(2.0, 1.0)) # 4.0 + 4.0ε
  echo sq(sq(D(2.0, 1.0))) # 16.0 + 32.0ε

  # Example for the Huber Loss.
  # Notice the case distinction!
  proc huber(x: Dual, delta=1.0): Dual =
    result = if abs(x) < delta:
               0.5 * x^2
             else:
               delta * (abs(x) - 0.5 * delta)

  echo huber(D(-0.2,1)) # 0.02 + -0.2ε
  echo huber(D(5,2)) # 4.5 + 2.0ε
  echo huber(D(-5,2)) # 4.5 + -2.0ε


  # Examples
  echo gradient(huber, -5) # -1.0
  import sequtils
  proc norm_squared(x: openArray[Dual]): Dual = sum(x.mapIt(it ^ 2))
  echo gradient(norm_squared, [3, 4]) # [6.0, 8.0]
