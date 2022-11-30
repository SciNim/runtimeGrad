import hashes, sets, math, strformat, algorithm, strutils
export math

type
  BackFn* = proc(r: Value)
  Value* = ref object ## make normal object, or do we get into trouble?
    data*: float
    grad*: float
    backFn*: BackFn
    prev*: HashSet[Value]
    op*: string

proc hash*(v: Value): Hash =
  result = hash(v.data)
  result = result !& hash(v.grad)
  result = result !& hash(v.backFn)
  result = result !& hash(v.prev)

proc pretty*(x: Value, precision: int): string =
  result = "(p: " & formatBiggestFloat(x.data, precision = precision) &
    ", ∂: " & formatBiggestFloat(x.grad, precision = precision) & ")"
proc `$`*(x: Value): string = pretty(x, precision = -1)

proc initValue*[T: SomeNumber](x: T, children = initHashSet[Value](), op = ""): Value =
  result = Value(data: x.float, grad: 0.0, prev: children, op: op,
                 backFn: (proc(r: Value) = discard))

proc toSet(args: varargs[Value]): HashSet[Value] =
  result = initHashSet[Value]()
  for arg in args:
    result.incl arg

proc `+`*[T](v: Value, w: T): Value =
  when T is SomeNumber:
    var val = initValue(w)
  elif T is Value:
    template val: untyped = w
  result = initValue(v.data + val.data, toSet(v, val), "+")
  result.backFn = (
    proc(r: Value) =
      v.grad += r.grad
      val.grad += r.grad
  )
proc `+`*[T: not Value](v: T, w: Value): Value = w + v

proc `*`*[T](v: Value, w: T): Value =
  when T is SomeNumber:
    let val = initValue(w)
  else:
    template val: untyped = w
  result = initValue(v.data * val.data, toSet(v, val), "*")
  result.backFn = (
    proc(r: Value) =
      v.grad += val.data * r.grad
      val.grad += v.data * r.grad
  )
proc `*`*[T: not Value](v: T, w: Value): Value = w * v

proc `**`*[T: SomeNumber; U: SomeNumber](v: T, val: U): float = result = pow(v.float, val.float)
proc `**`*[T: SomeNumber](v: Value, val: T): Value =
  result = initValue(pow(v.data, val.float), toSet(v), "**" & $val)
  result.backFn = (
    proc(r: Value) =
      v.grad += (val.float * pow(v.data, (val - 1).float)) * r.grad
  )

proc relu*(v: Value): Value =
  result = initValue(if v.data < 0: 0.0 else: v.data, toSet(v), "relu")
  result.backFn = (
    proc(r: Value) =
      if r.data > 0:
        v.grad += r.grad
  )

proc `^`*(x, y: float): float = pow(x, y)

template genCall(fn, xId, deriv: untyped): untyped =
  proc fn*(v: Value): Value =
    ## letStmt contains the definition of x, `let x = m.val`
    result = initValue(`fn`(v.data), toSet(v), astToStr(`fn`))
    result.backFn = (
      proc(r: Value) =
        let xId = v.data
        v.grad += deriv * r.grad
    )

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

proc backward*(v: Value) =
  var topo = newSeq[Value]()
  var visited = initHashSet[Value]()
  proc build_topo(w: Value) =
    if w notin visited:
      visited.incl w
      for ch in w.prev:
        build_topo(ch)
      topo.add w
  build_topo(v)
  v.grad = 1
  for w in topo.reversed():
    w.backFn(w) # this is ridiculous (having to give `w` as an argument)

proc `-`*(v: Value): Value = result = v * (-1)
proc `-`*[T](v: Value, w: T): Value = result = v + (-w)
proc `-`*[T: not Value](v: T, w: Value): Value = result = (-w) + v
proc `/`*[T: not Value](v: Value, w: T): Value = result = v * (w ** (-1.0))
proc `/`*[T](v: T, w: Value): Value = result = (w ** (-1.0)) * v
proc `+=`*[T](v: var Value, w: T) = v = v + w

when isMainModule:
  let a = initValue(-4.0)
  let b = initValue(2.0)
  var c = a + b
  var d = a * b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + (b + a).relu()
  d += 3 * d + (b - a).relu()
  let e = c - d
  let f = e**2
  var g = f / 2.0
  g += 10.0 / f
  echo &"{g.data:.4f}" # prints 24.7041, the outcome of this forward pass
  g.backward()
  echo &"{a.grad:.4f}" # prints 138.8338, i.e. the numerical value of dg/da
  echo &"{b.grad:.4f}" # prints 645.5773, i.e. the numerical value of dg/db

  let arg = initValue(0.5)
  let x = tanh(arg)
  x.backward()
  echo "AD: ∂(tanh(x))/∂x|_{x=0.5} = ", &"{arg.grad:.4f}"
  echo "∂(tanh(x))/∂x|_{x=0.5} = [1 / cosh²(x)]_{x=0.5} = ", &"{1.0 / (cosh(0.5)**2):.4f}"
