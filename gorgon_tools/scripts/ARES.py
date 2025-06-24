"""ARES: Auroral Rendering on Earth's Surface (JWBE 12/09/2022).

trace generated using paraview version 5.9.1 (converted manually to 5.8.0)
"""
import datetime as dt
import sys

import numpy as np
import paraview.simple as pvs

dat = np.genfromtxt(
    sys.argv[1] + "/Gorgon_axes.csv", delimiter=",", skip_header=1, dtype=np.float64
)
dat_nm = np.genfromtxt(sys.argv[1] + "/Gorgon_axes.csv", delimiter=",", dtype=str)[0, :]
dat_nm = np.char.strip(dat_nm)
itime = np.argwhere(dat_nm == "time")[0][0]
iGEO = np.argwhere(dat_nm == "GEO Z.x")[0][0]
iGSE = np.argwhere(dat_nm == "GSE X.x")[0][0]
it = np.argwhere(dat[:, itime] // 60 == float(sys.argv[2]) // 60)[0][0]
x, y, z = dat[it, iGEO], dat[it, iGEO + 1], dat[it, iGEO + 2]
colat = np.pi / 2 - np.arctan2(z, np.sqrt(x**2 + y**2))
lon = np.arctan2(x, y)
x, y, z = 0, colat * 180 / np.pi, lon * 180 / np.pi
S_x, S_y, S_z = dat[it, iGSE], dat[it, iGSE + 1], dat[it, iGSE + 2]

dt_str_format_in = "%Y-%m-%d_%H:%M:%S"  # Format of datettime read in by script
time = dt.datetime.strptime(sys.argv[3], dt_str_format_in) + dt.timedelta(
    seconds=int(sys.argv[2])
)
timestr = time.strftime("%Y/%m/%d %H:%M:%S")

MJD = (time - dt.datetime(1858, 11, 17)).days
T0 = (MJD - 51544.5) / 36525.0
UT = (time - dt.datetime(time.year, time.month, time.day)).total_seconds() / 3600
theta = 100.461 + 36000.770 * T0 + 15.04107 * UT

#### disable automatic camera reset on 'Show'
pvs._DisableFirstRenderCameraReset()

# Set-up render view
text1 = pvs.Text(registrationName="Text1")
text1.Text = timestr
renderView1 = pvs.GetActiveViewOrCreate("RenderView")
text1Display = pvs.Show(text1, renderView1, "TextSourceRepresentation")
renderView1.ResetCamera()
materialLibrary1 = pvs.GetMaterialLibrary()
renderView1.Update()
text1Display.FontSize = 2
text1Display.FontSize = 15
text1Display.FontFamily = "Courier"
text1Display.Bold = 0
text1Display.Shadow = 1

# Load ionospheric data
iS2400vtp = pvs.XMLPolyDataReader(
    registrationName="IS-" + sys.argv[2] + ".vtp",
    FileName=[sys.argv[1] + "/IS/IS-" + sys.argv[2] + ".vtp"],
)
iS2400vtp.PointArrayStatus = ["phi", "e- energy"]
iS2400vtpDisplay = pvs.Show(iS2400vtp, renderView1, "GeometryRepresentation")
phiLUT = pvs.GetColorTransferFunction("phi")
phiLUT.RGBPoints = [
    -46069.8,
    0.231373,
    0.298039,
    0.752941,
    59.64999999999418,
    0.865003,
    0.865003,
    0.865003,
    46189.09999999999,
    0.705882,
    0.0156863,
    0.14902,
]
phiLUT.ScalarRangeInitialized = 1.0
iS2400vtpDisplay.Representation = "Surface"
iS2400vtpDisplay.AmbientColor = [0.0, 0.0, 0.0]
iS2400vtpDisplay.ColorArrayName = ["POINTS", "phi"]
iS2400vtpDisplay.DiffuseColor = [0.0, 0.0, 0.0]
iS2400vtpDisplay.LookupTable = phiLUT
iS2400vtpDisplay.OSPRayScaleArray = "phi"
iS2400vtpDisplay.OSPRayScaleFunction = "PiecewiseFunction"
iS2400vtpDisplay.SelectOrientationVectors = "Evec"
iS2400vtpDisplay.ScaleFactor = 0.20345399379730225
iS2400vtpDisplay.SelectScaleArray = "phi"
iS2400vtpDisplay.GlyphType = "Arrow"
iS2400vtpDisplay.GlyphTableIndexArray = "phi"
iS2400vtpDisplay.GaussianRadius = 0.010172699689865113
iS2400vtpDisplay.SetScaleArray = ["POINTS", "phi"]
iS2400vtpDisplay.ScaleTransferFunction = "PiecewiseFunction"
iS2400vtpDisplay.OpacityArray = ["POINTS", "phi"]
iS2400vtpDisplay.OpacityTransferFunction = "PiecewiseFunction"
iS2400vtpDisplay.DataAxesGrid = "GridAxesRepresentation"
iS2400vtpDisplay.PolarAxes = "PolarAxesRepresentation"
iS2400vtpDisplay.ScaleTransferFunction.Points = [
    -46069.8,
    0.0,
    0.5,
    0.0,
    46189.09999999999,
    1.0,
    0.5,
    0.0,
]
iS2400vtpDisplay.OpacityTransferFunction.Points = [
    -46069.8,
    0.0,
    0.5,
    0.0,
    46189.09999999999,
    1.0,
    0.5,
    0.0,
]
materialLibrary1 = pvs.GetMaterialLibrary()
iS2400vtpDisplay.SetScalarBarVisibility(renderView1, True)
phiLUT.RescaleTransferFunction(-46069.8, 46189.1)
phiPWF = pvs.GetOpacityTransferFunction("phi")
phiPWF.Points = [-46069.8, 0.0, 0.5, 0.0, 46189.09999999999, 1.0, 0.5, 0.0]
phiPWF.ScalarRangeInitialized = 1
phiPWF.RescaleTransferFunction(-46069.8, 46189.1)
pvs.ColorBy(iS2400vtpDisplay, ("POINTS", "e- energy"))
pvs.HideScalarBarIfNotNeeded(phiLUT, renderView1)
iS2400vtpDisplay.RescaleTransferFunctionToDataRange(True, False)
iS2400vtpDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'eenergy'
eenergyLUT = pvs.GetColorTransferFunction("eenergy")
eenergyLUT.EnableOpacityMapping = 1
eenergyLUT.RGBPoints = [
    0.0,
    0.054902,
    0.109804,
    0.121569,
    1.0676,
    0.07451,
    0.172549,
    0.180392,
    2.1352,
    0.086275,
    0.231373,
    0.219608,
    3.2028,
    0.094118,
    0.278431,
    0.25098,
    4.2704,
    0.109804,
    0.34902,
    0.278431,
    5.338,
    0.113725,
    0.4,
    0.278431,
    6.4056,
    0.117647,
    0.45098,
    0.270588,
    7.473199999999999,
    0.117647,
    0.490196,
    0.243137,
    8.5408,
    0.113725,
    0.521569,
    0.203922,
    9.6084,
    0.109804,
    0.54902,
    0.152941,
    10.676,
    0.082353,
    0.588235,
    0.082353,
    11.7436,
    0.109804,
    0.631373,
    0.05098,
    12.8112,
    0.211765,
    0.678431,
    0.082353,
    13.8788,
    0.317647,
    0.721569,
    0.113725,
    14.946399999999999,
    0.431373,
    0.760784,
    0.160784,
    16.014,
    0.556863,
    0.8,
    0.239216,
    17.0816,
    0.666667,
    0.839216,
    0.294118,
    18.1492,
    0.784314,
    0.878431,
    0.396078,
    19.2168,
    0.886275,
    0.921569,
    0.533333,
    20.284399999999998,
    0.960784,
    0.94902,
    0.670588,
    21.352,
    1.0,
    0.984314,
    0.901961,
]
eenergyLUT.ColorSpace = "Lab"
eenergyLUT.NanColor = [0.25, 0.0, 0.0]
eenergyLUT.ScalarRangeInitialized = 1.0
eenergyLUTColorBar = pvs.GetScalarBar(eenergyLUT, renderView1)
eenergyLUTColorBar.WindowLocation = "AnyLocation"
eenergyLUTColorBar.Position = [0.77, 0.3404522613065327]
eenergyLUTColorBar.Title = r"e$^-$ energy / erg"
eenergyLUTColorBar.ComponentTitle = ""
eenergyLUTColorBar.ScalarBarLength = 0.3299999999999993
eenergyLUTColorBar.TitleFontSize = 12
eenergyLUTColorBar.LabelFontSize = 12
eenergyPWF = pvs.GetOpacityTransferFunction("eenergy")
eenergyPWF.Points = [0.0, 0.0, 0.5, 0.0, 21.352, 1.0, 0.5, 0.0]
eenergyPWF.ScalarRangeInitialized = 1

# Add Earth as sphere
sphere1 = pvs.Sphere(registrationName="Sphere1")
sphere1.Radius = 1.0
sphere1.ThetaResolution = 100
sphere1.StartTheta = 1e-05
sphere1.PhiResolution = 100

# Blue Marble texture for month of year from https://visibleearth.nasa.gov/
textureMaptoSphere1 = pvs.TextureMaptoSphere(
    registrationName="TextureMaptoSphere1", Input=sphere1
)
textureMaptoSphere1.PreventSeam = 0
textureMaptoSphere1Display = pvs.Show(
    textureMaptoSphere1, renderView1, "GeometryRepresentation"
)
textureMaptoSphere1Display.Representation = "Surface"
textureMaptoSphere1Display.AmbientColor = [0.0, 0.0, 0.0]
textureMaptoSphere1Display.DiffuseColor = [0.0, 0.0, 0.0]
textureMaptoSphere1Display.add_attribute("SelectTCoordArray", "Texture Coordinates")
textureMaptoSphere1Display.SelectTCoordArray = "Texture Coordinates"
textureMaptoSphere1Display.OSPRayScaleArray = "Normals"
textureMaptoSphere1Display.OSPRayScaleFunction = "PiecewiseFunction"
textureMaptoSphere1Display.ScaleFactor = 0.2
textureMaptoSphere1Display.GlyphType = "Arrow"
textureMaptoSphere1Display.GaussianRadius = 0.01
textureMaptoSphere1Display.SetScaleArray = ["POINTS", "Normals"]
textureMaptoSphere1Display.ScaleTransferFunction = "PiecewiseFunction"
textureMaptoSphere1Display.OpacityArray = ["POINTS", "Normals"]
textureMaptoSphere1Display.OpacityTransferFunction = "PiecewiseFunction"
textureMaptoSphere1Display.DataAxesGrid = "GridAxesRepresentation"
textureMaptoSphere1Display.PolarAxes = "PolarAxesRepresentation"
textureMaptoSphere1Display.Ambient = 0.2
textureMaptoSphere1Display.Opacity = 0.45
textureMaptoSphere1Display.ScaleTransferFunction.Points = [
    -0.9998741149902344,
    0.0,
    0.5,
    0.0,
    0.9998741149902344,
    1.0,
    0.5,
    0.0,
]
textureMaptoSphere1Display.OpacityTransferFunction.Points = [
    -0.9998741149902344,
    0.0,
    0.5,
    0.0,
    0.9998741149902344,
    1.0,
    0.5,
    0.0,
]
textureMaptoSphere1Display.MapScalars = 0
textureMaptoSphere1Display.Orientation = [x, y, z]
textureMaptoSphere1Display.PolarAxes.Orientation = [x, y, z]
outfile2 = pvs.CreateTexture(
    "/rds/general/user/je517/home/BlueMarble/" + time.strftime("%b") + "_Earth.png"
)
textureMaptoSphere1Display.Texture = outfile2
textureMaptoSphere1Display.AmbientColor = [1.0, 1.0, 1.0]
textureMaptoSphere1Display.DiffuseColor = [1.0, 1.0, 1.0]

# Night-time 'city lights' texture from https://visibleearth.nasa.gov/
textureMaptoSphere2 = pvs.TextureMaptoSphere(
    registrationName="TextureMaptoSphere2", Input=sphere1
)
textureMaptoSphere2.PreventSeam = 0
textureMaptoSphere2Display = pvs.Show(
    textureMaptoSphere2, renderView1, "GeometryRepresentation"
)
textureMaptoSphere2Display.Representation = "Surface"
textureMaptoSphere2Display.AmbientColor = [0.0, 0.0, 0.0]
textureMaptoSphere2Display.DiffuseColor = [0.0, 0.0, 0.0]
textureMaptoSphere2Display.add_attribute("SelectTCoordArray", "Texture Coordinates")
textureMaptoSphere2Display.SelectTCoordArray = "Texture Coordinates"
textureMaptoSphere2Display.OSPRayScaleArray = "Normals"
textureMaptoSphere2Display.OSPRayScaleFunction = "PiecewiseFunction"
textureMaptoSphere2Display.ScaleFactor = 0.2
textureMaptoSphere2Display.GlyphType = "Arrow"
textureMaptoSphere2Display.GaussianRadius = 0.01
textureMaptoSphere2Display.SetScaleArray = ["POINTS", "Normals"]
textureMaptoSphere2Display.ScaleTransferFunction = "PiecewiseFunction"
textureMaptoSphere2Display.OpacityArray = ["POINTS", "Normals"]
textureMaptoSphere2Display.OpacityTransferFunction = "PiecewiseFunction"
textureMaptoSphere2Display.DataAxesGrid = "GridAxesRepresentation"
textureMaptoSphere2Display.PolarAxes = "PolarAxesRepresentation"
textureMaptoSphere2Display.Ambient = 0.2
textureMaptoSphere2Display.Opacity = 1
textureMaptoSphere2Display.ScaleTransferFunction.Points = [
    -0.9998741149902344,
    0.0,
    0.5,
    0.0,
    0.9998741149902344,
    1.0,
    0.5,
    0.0,
]
textureMaptoSphere2Display.OpacityTransferFunction.Points = [
    -0.9998741149902344,
    0.0,
    0.5,
    0.0,
    0.9998741149902344,
    1.0,
    0.5,
    0.0,
]
textureMaptoSphere2Display.MapScalars = 0
textureMaptoSphere2Display.Orientation = [x, y, z]
textureMaptoSphere2Display.PolarAxes.Orientation = [x, y, z]
outfile3 = pvs.CreateTexture("/rds/general/user/je517/home/BlueMarble/Night_Earth.jpg")
textureMaptoSphere2Display.Texture = outfile3
textureMaptoSphere2Display.AmbientColor = [1.0, 1.0, 1.0]
textureMaptoSphere2Display.DiffuseColor = [1.0, 1.0, 1.0]

# Add celestial sphere
sphere2 = pvs.Sphere(registrationName="Sphere2")
sphere2.Radius = 200.0
sphere2.ThetaResolution = 100
sphere2.StartTheta = 1e-05
sphere2.PhiResolution = 100
sphere2Display = pvs.Show(sphere2, renderView1, "GeometryRepresentation")

# Full-sky star map from https://in-the-sky.org/
# textureMaptoSphere3 = TextureMaptoSphere(registrationName='TextureMaptoSphere3',
# Input=sphere2)
# textureMaptoSphere3.PreventSeam = 0
# textureMaptoSphere3Display = Show(textureMaptoSphere3, renderView1,
# 'GeometryRepresentation')
# textureMaptoSphere3Display.Representation = 'Surface'
# textureMaptoSphere3Display.AmbientColor = [1.0, 1.0, 1.0]
# textureMaptoSphere3Display.DiffuseColor = [1.0, 1.0, 1.0]
# textureMaptoSphere3Display.add_attribute('SelectTCoordArray','Texture Coordinates')
# textureMaptoSphere3Display.SelectTCoordArray = 'Texture Coordinates'
# textureMaptoSphere3Display.OSPRayScaleArray = 'Normals'
# textureMaptoSphere3Display.OSPRayScaleFunction = 'PiecewiseFunction'
# textureMaptoSphere3Display.ScaleFactor = 40
# textureMaptoSphere3Display.GlyphType = 'Arrow'
# textureMaptoSphere3Display.GaussianRadius = 2.0
# textureMaptoSphere3Display.SetScaleArray = ['POINTS', 'Normals']
# textureMaptoSphere3Display.ScaleTransferFunction = 'PiecewiseFunction'
# textureMaptoSphere3Display.OpacityArray = ['POINTS', 'Normals']
# textureMaptoSphere3Display.OpacityTransferFunction = 'PiecewiseFunction'
# textureMaptoSphere3Display.DataAxesGrid = 'GridAxesRepresentation'
# textureMaptoSphere3Display.PolarAxes = 'PolarAxesRepresentation'
# textureMaptoSphere3Display.Ambient = 0
# textureMaptoSphere3Display.Opacity = 1
# textureMaptoSphere3Display.ScaleTransferFunction.Points = [-0.9998741149902344, 0.0,
# 0.5, 0.0, 0.9998741149902344, 1.0, 0.5, 0.0]
# textureMaptoSphere3Display.OpacityTransferFunction.Points = [-0.9998741149902344, 0.0,
# 0.5, 0.0, 0.9998741149902344, 1.0, 0.5, 0.0]
# textureMaptoSphere3Display.MapScalars = 0
# textureMaptoSphere3Display.Orientation = [x, y, z+theta]
# textureMaptoSphere3Display.PolarAxes.Orientation = [x, y, z+theta]
# outfile4 = CreateTexture('/rds/general/user/je517/home/BlueMarble/Star_Sphere.png')
# textureMaptoSphere3Display.Texture = outfile4
# Hide(sphere2, renderView1)

# renderView1.Update()
textureMaptoSphere3 = pvs.TextureMaptoSphere(
    registrationName="TextureMaptoSphere3", Input=sphere2
)
textureMaptoSphere3.PreventSeam = 0
textureMaptoSphere3Display = pvs.Show(
    textureMaptoSphere3, renderView1, "GeometryRepresentation"
)
textureMaptoSphere3Display.Representation = "Surface"
textureMaptoSphere3Display.AmbientColor = [1.0, 1.0, 1.0]
textureMaptoSphere3Display.ColorArrayName = [None, ""]
textureMaptoSphere3Display.DiffuseColor = [1.0, 1.0, 1.0]
textureMaptoSphere3Display.OSPRayScaleArray = "Normals"
textureMaptoSphere3Display.OSPRayScaleFunction = "PiecewiseFunction"
textureMaptoSphere3Display.ScaleFactor = 40.0
textureMaptoSphere3Display.GlyphType = "Arrow"
textureMaptoSphere3Display.GaussianRadius = 2.0
textureMaptoSphere3Display.SetScaleArray = ["POINTS", "Normals"]
textureMaptoSphere3Display.ScaleTransferFunction = "PiecewiseFunction"
textureMaptoSphere3Display.OpacityArray = ["POINTS", "Normals"]
textureMaptoSphere3Display.OpacityTransferFunction = "PiecewiseFunction"
textureMaptoSphere3Display.DataAxesGrid = "GridAxesRepresentation"
textureMaptoSphere3Display.PolarAxes = "PolarAxesRepresentation"
textureMaptoSphere3Display.ScaleTransferFunction.Points = [
    -0.9998741149902344,
    0.0,
    0.5,
    0.0,
    0.9998741149902344,
    1.0,
    0.5,
    0.0,
]
textureMaptoSphere3Display.OpacityTransferFunction.Points = [
    -0.9998741149902344,
    0.0,
    0.5,
    0.0,
    0.9998741149902344,
    1.0,
    0.5,
    0.0,
]
pvs.Hide(sphere2, renderView1)
# renderView1.Update()
star_Sphere = pvs.CreateTexture(
    "/rds/general/user/je517/home/BlueMarble/Star_Sphere.png"
)
textureMaptoSphere3Display.Texture = star_Sphere
renderView1.ResetCamera()

# Add solar illumination
light = pvs.AddLight(view=renderView1)
light.Position = [50 * S_x, 50 * S_y, 50 * S_z]
light.Intensity = 4.0

renderView1.Update()

renderView1.ResetCamera()


#### disable automatic camera reset on 'Show'
pvs._DisableFirstRenderCameraReset()

# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# get layout
layout1 = pvs.GetLayout()

# --------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize([1140, 750])

# -----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [2.8, 1.0, 3.7]
renderView1.CameraFocalPoint = [0.06882940464972598, 0.25, 0.2]
renderView1.CameraViewUp = [-0.7498611547669278, 0.01524454202263308, 0.661419573727698]
renderView1.CameraParallelScale = 400

# --------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).

pvs.SaveScreenshot(
    sys.argv[1] + "/Plots/IS/ARES_" + time.strftime("%Y-%m-%d_%H_%M") + ".jpg",
    renderView1,
    ImageResolution=[1500, 1090],
)
