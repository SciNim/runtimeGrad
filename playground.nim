import sequtils, stats

## Choose eithe Micrograd or Duals to run the code with either module
const Micrograd {.booldefine.} = false
const Duals {.booldefine.} = false
const Complex {.booldefine.} = true

when Micrograd:
  import micrograd
when Duals:
  import dual_autograd
when Complex:
  import complex_step

const PITCH = 0.055
proc eccentricity[T](θ: T, c: seq[(int, int)], centerX, centerY: float): T =
  ## this function calculates the eccentricity of a found pixel cluster
  when Micrograd:
    var
      sum_x  = initValue(0.0)
      sum_y  = initValue(0.0)
      sum_x2 = initValue(0.0)
      sum_y2 = initValue(0.0)
  when Duals:
    var
      sum_x  = D(0.0, 0.0)
      sum_y  = D(0.0, 0.0)
      sum_x2 = D(0.0, 0.0)
      sum_y2 = D(0.0, 0.0)
  when Complex:
    var
      sum_x  = cmplx(0.0)
      sum_y  = cmplx(0.0)
      sum_x2 = cmplx(0.0)
      sum_y2 = cmplx(0.0)

  for i in 0 ..< len(c):
    let
      new_x = cos(θ) * (c[i][0].float - centerX) * PITCH - sin(θ) * (c[i][1].float - centerY) * PITCH
      new_y = sin(θ) * (c[i][0].float - centerX) * PITCH + cos(θ) * (c[i][1].float - centerY) * PITCH
    sum_x += new_x
    sum_y += new_y
    sum_x2 += (new_x * new_x)
    sum_y2 += (new_y * new_y)

  let
    n_elements = len(c).float
    rms_x = sqrt( (sum_x2 / n_elements) - (sum_x * sum_x / n_elements / n_elements))
    rms_y = sqrt( (sum_y2 / n_elements) - (sum_y * sum_y / n_elements / n_elements))

  # calc eccentricity from RMS
  let exc = rms_x / rms_y
  # we want to maximize the eccentricity, hence we return the negative eccentricity,
  # so that gradient descent finds the largest minimal value
  result = -exc


## Define some (x / y) coordinates (these come from a gaseous detector with a Timepix ASIC based readout;
## Each hit pixel corresponds to a primary electron produced by ionization of an event in the soft X-ray
## energy range.)
let pixels = @[(221, 144), (222, 139), (223, 139), (237, 138), (217, 137), (226, 137), (222, 136), (224, 135), (203, 134), (225, 133), (236, 133), (226, 131), (228, 131), (216, 130), (212, 129), (237, 129), (227, 127), (237, 125), (187, 124), (188, 124), (231, 124), (218, 123), (228, 123), (233, 123), (200, 122), (223, 122), (224, 122), (185, 121), (223, 121), (219, 120), (200, 119), (214, 119), (179, 118), (203, 117), (209, 117), (233, 117), (195, 115), (211, 115), (241, 115), (212, 114), (214, 114), (218, 114), (209, 113), (219, 113), (221, 113), (232, 113), (234, 113), (240, 113), (178, 112), (179, 112), (203, 112), (223, 112), (230, 112), (231, 112), (237, 112), (240, 112), (201, 111), (226, 111), (233, 111), (240, 111), (241, 111), (197, 110), (215, 110), (220, 110), (227, 110), (246, 110), (190, 109), (203, 109), (207, 109), (213, 109), (237, 109), (186, 108), (192, 108), (199, 108), (178, 107), (188, 107), (203, 107), (208, 107), (209, 107), (215, 107), (216, 107), (220, 107), (247, 107), (204, 106), (210, 106), (218, 106), (237, 106), (240, 106), (183, 105), (212, 105), (217, 105), (224, 105), (226, 105), (230, 105), (231, 105), (234, 105), (241, 105), (175, 104), (176, 104), (189, 104), (227, 104), (230, 104), (237, 104), (201, 103), (217, 103), (220, 103), (226, 103), (232, 103), (191, 102), (203, 102), (204, 102), (205, 102), (210, 102), (219, 102), (222, 102), (195, 101), (204, 101), (206, 101), (217, 101), (225, 101), (178, 100), (197, 100), (208, 100), (217, 100), (226, 100), (228, 100), (230, 100), (232, 100), (233, 100), (236, 100), (243, 100), (246, 100), (207, 99), (209, 99), (211, 99), (217, 99), (197, 98), (199, 98), (203, 98), (204, 98), (205, 98), (209, 98), (210, 98), (212, 98), (214, 98), (217, 98), (218, 98), (226, 98), (229, 98), (231, 98), (236, 98), (197, 97), (213, 97), (221, 97), (222, 97), (231, 97), (235, 97), (239, 97),(213, 96), (214, 96), (220, 96), (226, 96), (229, 96), (236, 96), (242, 96), (222, 95), (223, 95), (226, 95), (227, 95), (228, 95), (229, 95), (191, 94), (206, 94), (207, 94), (210, 94), (215, 94), (225, 94), (210, 93), (213, 93), (214, 93), (230, 93), (240, 93), (204, 92), (209, 92), (222, 92), (223, 92), (193, 91), (196, 91), (214, 91), (220, 91), (226, 91), (231, 91), (236, 91), (237, 91), (203, 90), (206, 90), (209, 90), (216, 90), (220, 90), (222, 90), (210,89), (213, 89), (217, 89), (219, 89), (224, 89), (226, 89), (228, 89), (229, 89), (243, 89), (187, 88), (209, 88), (216, 88), (225, 88), (179, 88), (180, 88), (184, 87), (188, 87), (219, 87), (227, 86), (228, 86), (231, 86), (198, 85), (199, 85), (206, 85), (207, 85), (208, 85), (213, 85), (214, 85), (218, 85), (221, 85), (225, 85), (230, 85), (244, 85), (197, 84), (200, 84), (226, 84), (229, 84), (230, 84), (239, 84), (245, 84), (196, 83), (217, 83), (220, 83), (237, 83), (199, 82), (200, 82), (201, 82), (203, 82), (204, 82), (222, 82), (223, 82), (235, 82), (242, 82), (205, 81), (206, 81), (208, 81), (210, 81), (214, 81), (215, 81), (216, 81), (217, 81), (218, 81), (242, 81), (248, 81), (213, 80), (216, 80), (222, 80), (229, 80), (233, 80), (234, 80), (235, 80), (239, 80), (248, 80), (178, 79), (231, 79),(235, 79), (237, 79), (238, 79), (178, 78), (204, 78), (208, 78), (213, 78), (222, 78), (212, 77), (213, 77), (218, 77), (232, 77), (243, 77), (244, 77), (203, 76), (209, 76), (221, 76), (227, 76), (228, 76), (229, 76), (197, 75), (225, 75), (234, 75), (248, 75), (186, 76), (214, 74), (233, 74), (200, 73), (214, 73), (217, 73), (238, 73), (173, 72), (206, 72), (207, 72), (220, 72), (245, 72), (212, 71), (219, 71), (233, 71), (199, 70), (211, 70), (225, 70), (214,69), (229, 69), (247, 69), (198, 68), (225, 68), (231, 68), (232, 68), (209, 67), (221, 67), (224, 67), (237, 67), (246, 67), (214, 66), (185, 65), (197, 65), (207, 65), (195, 64), (208, 64), (218, 64), (221, 64), (234, 64), (236, 64), (199, 62), (211, 62), (229, 62), (192, 61), (243, 61), (244, 61), (183, 60), (233, 60), (223, 58), (226, 58), (214, 57), (225, 57), (225, 55), (218, 53), (229, 53), (195, 51), (206, 51), (218, 50), (206, 46), (237, 45), (208, 44), (209, 43), (234, 40), (216, 24)]

let rotAngle = 1.530112032335672
let ecc      = -1.209781225615816
## compute the center position of the X and Y coordinates each
let (centerX, centerY) = (pixels.mapIt(it[0]).mean, pixels.mapIt(it[1]).mean)

## Now perform the gradient descent (either using `micrograd` or `dual_autograd`
when Micrograd:
  var θp = 1.1 # pick some starting angle (in radian)
  var θ = initValue(θp) # and create a `Value` from it
  var res = eccentricity(θ, pixels, centerX, centerY) # propagate through eccentricity calculation
  res.backward() # use DAG to compute backward pass (so that θ.grad is non zero)
  var i = 0
  while abs(0.1 * θ.grad) > 1e-8: # gradient descent until next iteration would change
                                  # less than some epsilon
    θp = θp - 0.1 * θ.grad        # update the gradient
    θ = initValue(θp)             # recreate a new value
    res = eccentricity(θ, pixels, centerX, centerY)
    res.backward()                # and calculate graph & use it
    echo "i = ", i, "\n\tθ = ", θ, " ε = ", res.data
    inc i

  echo "Final result:"
  echo "\tθ = ", pretty(θ, precision = 16), " ε = ", res.data
  echo "\tExpected: θ = ", rotAngle, " ε = ", ecc
  ## As we can see in the output
  ##   Final result:
  ##         θ = (p: 1.530111927499925, ∂: -9.800937286362152e-08) ε = -1.20978122561581
  ##         Expected: θ = 1.530112032335672 ε = -1.209781225615816
  ## the expected values (computed using non linear optimization with `nlopt`) and our
  ## naive gradient descent approach give the same result within the expectation given
  ## our 1e-8 epsilon.

## The Dual based code is the same as above. The only difference is the source of the
## relevant gradient `∂f/∂θ` (here it falls out as the gradient of the resulting `Dual`
## returned from the `eccentricity` call, whereas in the above it is the gradient
## associated with the input ref object `θ`.
when Duals:
  var θp = 1.1
  var θ = D(θp, 1.0)

  var res = eccentricity(θ, pixels, centerX, centerY)
  echo res
  var i = 0
  while abs(0.1 * res.d) > 1e-8:
    θp = θp - 0.1 * res.d
    θ = D(θp, 1.0)
    res = eccentricity(θ, pixels, centerX, centerY)
    echo "i = ", i, "\n\tθ = ", θ.p, " ε = ", res
    inc i
  echo "Final result:"
  echo "\tθ = ", θ.p, " ε = ", pretty(res, precision = 16)
  echo "\tExpected: θ = ", rotAngle, " ε = ", ecc
  ## And the results from the dual based code:
  ##   Final result:
  ##         θ = 1.530111927499925 ε = (p: -1.209781225615810, ∂: -9.800937291611005e-08)
  ##         Expected: θ = 1.530112032335672 ε = -1.209781225615816

when Complex:
  var θp = 1.1
  var θ = cmplx(θp)

  var grad = derivative(θ, eccentricity(θ, pixels, centerX, centerY))
  var i = 0
  while abs(0.1 * grad) > 1e-8:
    θp = θp - 0.1 * grad
    θ = cmplx(θp)
    grad = derivative(θ, eccentricity(θ, pixels, centerX, centerY))
    echo "i = ", i, "\n\tθ = ", θ.re, " ε = ", grad
    inc i
  echo "Final result:"
  echo "\tθ = ", θ.re, " ε = ", eccentricity(θ, pixels, centerX, centerY), " grad = ", grad
  echo "\tExpected: θ = ", rotAngle, " ε = ", ecc
  ## And the results from the complex step derivative code:
  ##   Final result:
  ##           θ = 1.530111927499925 ε = (-1.20978122561581, -0.0) grad = -9.800937292803663e-08
  ##           Expected: θ = 1.530112032335672 ε = -1.209781225615816
