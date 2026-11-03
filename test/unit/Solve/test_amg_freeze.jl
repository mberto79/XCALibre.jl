@test BoomerAMG(freeze=7).freeze == 7 && AMG(coarsening=SmoothAggregation(), coarse_refresh_interval=7).coarse_refresh_interval == 7
@test BoomerAMG().freeze == 10 && AMG(coarsening=SmoothAggregation()).coarse_refresh_interval == 5
