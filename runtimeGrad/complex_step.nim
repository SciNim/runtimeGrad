import std / [math, strutils]
export math

## This implementation is based on the ideas of the paper:
## The Complex-Step Derivative Approximation, 2003
## by Martins, Sturdza, Alonso
## https://doi.org/10.1145/838250.838251

import std / complex except abs, `==`
export complex

const h_eps* = 1e-20

proc cmplx*[T](x: T): Complex[T] = Complex[T](re: x, im: T(0.0))
proc cmplx*[T](x: T, eps: T): Complex[T] = Complex[T](re: x, im: T(eps))

proc abs*[T](z: Complex[T]): Complex[T] =
  result = if z.re < 0: Complex[T](re: -z.re, im: -z.im)
           else: z
# helpers for convenience for floats
proc `**`*(x, y: float): float = pow(x, y)
proc `^`*(x, y: float): float = pow(x, y)

proc `<`*[T](z1, z2: Complex[T]): bool = z1.re < z2.re
proc `==`*[T](z1, z2: Complex[T]): bool = z1.re == z2.re
proc `**`*[T](z: Complex[T], k: T): Complex[T] = pow(z, k)
proc `^`*[T](z: Complex[T], k: T): Complex[T] = pow(z, k)

import macros
macro patchCall*(f, it, by: typed): untyped =
  doAssert by.kind in {nnkSym, nnkIdent} and it.kind in {nnkSym, nnkIdent}
  proc patchImpl(n: NimNode, it, by: string): NimNode =
    case n.kind
    of nnkSym, nnkIdent: result = if n.strVal == it: ident(by) else: n
    else:
      result = n.copy()
      for i in 0 ..< n.len:
        result[i] = patchImpl(n[i], it, by)
  result = patchImpl(f, it.strVal, by.strVal)

template derivative*[T](x: Complex[T], f: typed): T =
  ## Compute complex step derivative approximation:
  ##
  ## ∂f/∂x = Im[ f(x + ih) ] / h
  ##
  ## `f` must be a function call using the argument `x`.
  ##
  ## Performs convenient replacement of the argument `x` in
  ## the call `f`.
  var mz = x
  mz.im = h_eps # set starting h
  let res = f.patchCall(x, mz)
  (res / T(h_eps)).im
