import { Grid, Tooltip } from '@material-ui/core';
import React, { Component, FC } from 'react';
import ColorButton from 'compDir/Button';
import { TimePicker } from 'compDir/DateTimePicker';
import MDIcon from '@mdi/react';
import { mdiClockOutline, mdiPlus, mdiTrashCanOutline } from '@mdi/js';
import { Moment } from 'moment';
import { LabelOutline } from 'compDir/Outline';
import CustomLabel from 'compDir/CustomLabel';
import { IDeadline } from 'interfaceDir';

export interface IStateToProps {
	deadlines?: Array<IDeadline>
}

export interface IDispatchToProps {
	initData(_comp: Component): any
	onAdd(): any
	onChange(order: number, time: Moment): any
	onDelete(frequency: Frequency): any
}

interface IProps extends IDispatchToProps, IStateToProps {
}

const DeadlinePicker: FC<IProps> = (props: IProps) => {

	const deadlines = props.deadlines;
	const deadlinesSize = deadlines ? deadlines.length : 0;
	return (
		<LabelOutline label={<CustomLabel value='締切時刻' icon={mdiClockOutline} isRequired />} isError={false}>
			<Grid container spacing={1} direction='column' >
				<Grid item container spacing={1} >
					{
						deadlinesSize > 0 && deadlines.map((deadline, idx) => (
							<Grid key={idx} item container>
								<Grid item xs={10}>
									<TimePicker hour={deadline.hour} minute={deadline.minute}
										onChange={(time) => props.onChange(deadline.order, time)} />
								</Grid>
								<Grid item xs={2}>
									<Tooltip title={deadlinesSize < 2 ? '' : '削除'}>
										<ColorButton btnType="icon" btnColor="red" size='medium'
											{...deadlinesSize < 2 ? {
												disabled: true, onClick: undefined
											} : {
												onClick: () => props.onDelete(deadline.order)
											}}>
											<MDIcon size={'18px'} path={mdiTrashCanOutline} />
										</ColorButton>
									</Tooltip>
								</Grid>
							</Grid>
						))
					}
				</Grid>
				<Grid container item spacing={1} >
					<Grid item xs={10}>
						<ColorButton btnColor='green' fullWidth size='small' variant='outlined'
							startIcon={<MDIcon size={'18px'} path={mdiPlus} />}
							onClick={props.onAdd}>追加</ColorButton>
					</Grid>
					<Grid item xs={2}></Grid>
				</Grid>
			</Grid>
		</LabelOutline>
	);
};

export default DeadlinePicker;