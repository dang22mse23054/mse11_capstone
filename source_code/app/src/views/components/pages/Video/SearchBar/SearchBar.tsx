import React, { Component, Fragment } from 'react';
import MDIcon from '@mdi/react';
import { mdiMagnify } from '@mdi/js';
import { Box, Grid, FormControlLabel, TextField, Switch } from '@material-ui/core';
import ColorButton from 'compDir/Button';
import DropDownSelect from 'compDir/DropDownSelect';
import SingleLvTagInput from 'compDir/TagInput/SingleLevel';
import ProcMasterTagInput from 'compDir/TagInput/ProcMaster';
import { ITaskSearchOpt, ICategoryOption, IProcMaster, ICategory, IHoliday } from 'interfaceDir';
import { DatePicker } from 'compDir/DateTimePicker';
import { VideoStatus, TIME_ZONE, Status } from 'constDir';

import moment from 'moment-timezone';
moment.tz.setDefault(TIME_ZONE);

export interface IStateToProps {
	categoryList: Array<ICategory>
}

export interface IDispatchToProps {
}

interface IProps extends IDispatchToProps, IStateToProps {
	onSearch(): any
}

interface IState {
	advSearch: boolean
	searchOpts: ITaskSearchOpt
}

class SearchBar extends Component<IProps, IState> {

	// Set default properties's values
	public static defaultProps: IProps = {
	}

	// Set default state
	public state: IState = {
		advSearch: false,
		searchOpts: {
			keyword: '',
			status: VideoStatus.PLAYING.value,
			// startDate: moment().startOf('day'),
			// endDate: moment().endOf('day'),
			startDate: null,
			endDate: null,
		},
	}

	onChangeKeyword = (event: React.ChangeEvent<HTMLInputElement>) => {
		const value = (event.target as HTMLInputElement).value;
		this.setState({
			searchOpts: {
				...this.state.searchOpts,
				keyword: value
			}
		});
	}

	onChangeSearchStartDate = (startDate: moment.Moment) => {
		let { endDate } = this.state.searchOpts;

		if (startDate && endDate && startDate.isValid() && startDate > endDate) {
			endDate = moment(startDate).endOf('day');
		}

		this.setState({
			searchOpts: {
				...this.state.searchOpts,
				startDate,
				endDate
			}
		});
	}

	onChangeSearchEndDate = (endDate: moment.Moment) => {
		let { startDate } = this.state.searchOpts;

		if (endDate && endDate.isValid() && startDate > endDate) {
			startDate = moment(endDate).startOf('day');
		}
		this.setState({
			searchOpts: {
				...this.state.searchOpts,
				startDate,
				endDate
			}
		});
	}

	onChangeStatus = (selectedItem) => {
		this.setState({
			searchOpts: {
				...this.state.searchOpts,
				status: selectedItem ? selectedItem.value : null
			}
		});
	}

	onChangeCategory = (chips: Array<any>, action) => {
		this.setState({
			searchOpts: {
				...this.state.searchOpts,
				categories: chips.filter(item => item.value.status != Status.DELETED)?.map(item => parseInt(item.value.id))
			}
		});
	}

	onEnter = (e) => {
		if (e.which === 13 && !this.props.isLoading) {
			this.props.onSearch(e);
		}
	};

	public render(): React.ReactNode {
		const { searchOpts } = this.state;
		const destType = searchOpts.destType;

		return (
			<Grid container justify='space-between' alignItems='flex-start'>
				<Grid item style={{ flex: 6 }}>
					<Grid container spacing={3} justify='flex-end' >
						<Grid item style={{ flex: 1 }}>
							<Grid container spacing={2}>
								<Grid item container spacing={2}>
									<Grid item style={{ flex: 1 }}>
										<TextField fullWidth label="Keyword" type="search"
											// placeholder='Title'
											value={searchOpts.keyword}
											onChange={this.onChangeKeyword}
											onKeyPress={this.onEnter}
										/>
									</Grid>
									<Grid item xs={2}>
										{
											<DatePicker label="From Date"
												inputVariant={false}
												type='startDate'
												holidays={[]}
												onChange={this.onChangeSearchStartDate}
												onKeyPress={this.onEnter}
												value={searchOpts.startDate}
											/>
										}
									</Grid>
									<Grid item xs={2}>
										{
											<DatePicker label="To Date"
												inputVariant={false}
												type='endDate'
												holidays={[]}
												onChange={this.onChangeSearchEndDate}
												onKeyPress={this.onEnter}
												value={searchOpts.endDate}
											/>
										}
									</Grid>
									<Grid item style={{ flex: 0.8 }}>
										<SingleLvTagInput label='Category' limitTags={1}
											onChange={this.onChangeCategory}
											options={this.props.categoryList}
										/>
									</Grid>
									<Grid item xs={2}>
										<DropDownSelect label={'Status'}
											mustSelect={false}
											disabled={false}
											options={[VideoStatus.PLAYING, VideoStatus.PAUSED, VideoStatus.STOPPED]}
											value={searchOpts.status}
											onChange={this.onChangeStatus}
										/>
									</Grid>
								</Grid>
							</Grid>

						</Grid>
						<Grid item>
							<Grid container spacing={1} direction='column' alignItems='center'>
								<Grid item>
									<ColorButton variant='contained' btnColor='secondary'
										disabled={this.props.isLoading == true}
										startIcon={<MDIcon size={'20px'} path={mdiMagnify} />}
										onClick={this.props.onSearch}>
										Search
									</ColorButton>
								</Grid>
							</Grid>
						</Grid>
					</Grid >
				</Grid >
			</Grid >
		);
	}
}
export default SearchBar;