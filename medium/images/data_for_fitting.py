def data_for_fitting(*, building_id, date):
    """
    Retrieves data for fitting from the previous business day
    taking into account holidays
    """

    lease_start = None
    while lease_start is None:
        # Previous business day according to Pandas (might be a holiday)
        previous_bday = pd.to_datetime(date) - BDay(1)

        # If a holiday, this will return None
        lease_start = (db().execute(building_daily_stats.select().where(
            building_daily_stats.c.building_id == building_id).where(
                building_daily_stats.c.date == previous_bday)).fetchone().
                       lease_obligations_start_at)

        date = previous_bday

    # Retrieve 8 hours of data from the lease start
    return load_sensor_values(
        building_id=building_id,
        start_time=lease_start,
        end_time=lease_start + timedelta(hours=8))