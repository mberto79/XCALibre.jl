@test BoomerAMG(freeze=7).freeze == 7 && GAMG(freeze=7).freeze == 7
@test BoomerAMG().freeze == 10 && GAMG().freeze == 25
