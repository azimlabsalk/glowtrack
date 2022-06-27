def list_model(model_class, session):
    from columnar import columnar
    # from tabulate import tabulate

    if hasattr(model_class, 'display_attrs'):
        columns = model_class.display_attrs()
    else:
        columns = [col.name for col in model_class.__table__.columns]

    objects = session.query(model_class).all()
    rows = []
    for obj in objects:
        row = [getattr(obj, column) for column in columns]
        rows.append(row)

    table = columnar(rows, columns, no_borders=True)
    # table = tabulate(rows, headers=columns)

    print(table)
