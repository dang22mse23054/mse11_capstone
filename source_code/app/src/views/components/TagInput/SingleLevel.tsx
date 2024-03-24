
import React, { Component, Fragment } from 'react';
import { Grid, Chip, Tooltip } from '@material-ui/core';
import MDIcon from '@mdi/react';
import ReactAutosuggest, { IProps as SuperIProps, IState as SuperIState } from 'compDir/TagInput';
import { grey } from '@material-ui/core/colors';

interface IState extends SuperIState {
}

interface IProps extends SuperIProps {
	classes?: any
	onChange?(chips, reason): any
	group?: {
		propName: string
		optLabel: string
	}
	isError?: boolean
	errIcon?: boolean
	helperText?: string
}

class SingleLvTagInput extends Component<IProps, IState> {

	constructor(props) {
		super(props);
	}

	public static defaultProps: IProps = {
		isError: false,
		helperText: '',
		group: undefined,
	}

	public render = (): React.ReactNode => {
		const { group, ...otherProps } = this.props;

		return (
			<Fragment>
				<ReactAutosuggest fullWidth {...otherProps}
					// option format 
					// [
					// 	{ key: '1', value: { icon: mdiCalendar, name: 'abc' } },
					// 	{ key: '2', value: { icon: mdiCalendarMonth, name: 'def' } },
					// ]

					freeSolo={false}

					groupBy={group ? (option) => option.value[group.propName] : undefined}
					getOptionLabel={group ? (option) => option.value[group.optLabel] : undefined}

					renderTags={group ? (
						(values: Array<ICategoryOption>, getTagProps) => {

							return values.map((option, index) => {
								if (option) {
									return (
										<Chip key={`tag_${index}`} size='small' variant='default' style={{ borderColor: grey[500] }}
											onDelete={() => true}
											label={(
												<Tooltip title={
													<Fragment>
														<b>{option.value[group.propName]}</b><br />
														&nbsp;&nbsp;⎩ {option.value[group.optLabel]}<br />
													</Fragment>
												}>
													<span style={{ cursor: 'pointer' }}>{option.value[group.optLabel]}</span>
												</Tooltip>
											)}
											{...getTagProps({ index })}
										/>
									);
								}
								return false;
							});
						}
					) : (
						(values: any /* Array<IDataSource> */, getTagProps) => {
							return values.map((option, index) => {
								if (option) {
									return (
										<Chip key={index} size='small' label={option.value.name}
											icon={option.value.icon ? <MDIcon size={'11pt'} path={option.value.icon} /> : undefined}
											style={{
												borderColor: grey[600]
											}}
											onDelete={() => true}
											{...getTagProps({ index })}
										/>
									);
								}
								return false;
							});
						}
					)}

					renderOption={(option) => {
						return (
							<Grid container spacing={1}>
								{
									option.value.icon && (
										<Grid container item alignItems='center' style={{ width: 'auto' }}>
											<MDIcon size={'13pt'} path={option.value.icon} />
										</Grid>
									)
								}
								<Grid item>{option.value.name}</Grid>
							</Grid>
						);
					}}

				/>
			</Fragment>
		);
	}
}

export default SingleLvTagInput;



