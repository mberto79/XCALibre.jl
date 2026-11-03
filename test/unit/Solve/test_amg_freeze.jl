using XCALibre, Test

@test BoomerAMG(refresh=7).refresh == 7 && AMG(SmoothedAggregation(); refresh=7).refresh == 7
@test BoomerAMG().refresh == 10 && AMG(SmoothedAggregation()).refresh == 5
