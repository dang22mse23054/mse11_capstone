
import React, { Component, Fragment } from 'react';
import { mdiFlare } from '@mdi/js';
import { createStyles, Theme, withStyles } from '@material-ui/core/styles';
import { Grid, Tooltip, Chip, List, ListItem, ListSubheader, Box } from '@material-ui/core';
import MDIcon from '@mdi/react';
import { grey, indigo } from '@material-ui/core/colors';
import { IOption, MultiLevelParser } from 'servDir/autocomplete';
import ReactAutosuggest, { IProps as SuperIProps, IState as SuperIState } from 'compDir/TagInput';

interface IState extends SuperIState {
	options: Array<IOption>
	optionMap: Map<number, IOption>
	value: Array<IOption>
	isInit: boolean
}

interface IProps extends SuperIProps {
	mainGroupIcon?: any
	classes?: any
	mainGroupIcon?: any
	level?: number
	// GraphQL remote api
	remoteUrl?: string
	gqlMethod: string
	queryString: string
	componentName?: string
	isError?: boolean
	helperText?: string
}

interface GroupInfo {
	name: string
	current?: any
	children: Array<any>
}

const useStyles = (theme: Theme) => createStyles({
	mainGroupHeader: {
		paddingBlock: 6,
		paddingInline: 16,
		zIndex: 1000,
		'&[data-focus="true"]': {
			backgroundColor: `${grey[100]} !important`,
		}
	},
	mainGroupHeaderContent: {
		height: 25,
	},
	groupBox: {
		backgroundColor: `${theme.palette.background.paper} !important`,
		width: '100%'
	},
	subGroupHeader: {
		paddingBlock: 6,
		paddingInline: 25,
		'&[data-focus="true"]': {
			backgroundColor: `${grey[100]} !important`,
		},
	}
});

class MultiLvTagInput extends Component<IProps, IState> {

	constructor(props) {
		super(props);
	}

	public static defaultProps: IProps = {
		id: 'default_id',
		label: 'Valid values',
		// placeholder: 'キーワードを入力してください',
		variant: 'standard',
		size: 'medium',
		isError: false,
		helperText: '',
		disabled: false,
		// noOptionsText: 'No data!',
		level: 0,
		value: [],
		// options: [],
		limitTags: -1,
		componentName: 'MultiLvTagInput',
	}

	// Set default state
	public state: IState = {
		isInit: true
	}

	initValuePreProcessOptions = async (opts) => {
		if (opts) {
			const newOpts = new MultiLevelParser(opts).toAutocompleteObj(this.props.level);

			const optionMap = new Map<number, IOption>();
			newOpts.map(option => {
				optionMap.set(option.value.id, option);
			});


			// init value start
			const value = this.props.value;
			// find option belong to value
			const parsedValue: Array<IOption> = [];
			value.map((item) => {
				let key = '';
				if (item) {
					if (item.key) {
						key = item.key;
					} else if (item.id) {
						key = item.id;
					} else if (item.value?.id) {
						key = item.value.id;
					} else {
						key = item;
					}
				}
				const val = optionMap.get(`${key}`);
				if (val) {
					parsedValue.push(val);
				}
			});
			// init value end

			await this.setState({
				options: newOpts,
				optionMap,
				value: parsedValue
			});

			return newOpts;
		}
		return null;
	}

	preProcessOptions = async (opts) => {
		if (opts) {
			const newOpts = new MultiLevelParser(opts).toAutocompleteObj(this.props.level);

			const optionMap = new Map<number, IOption>();
			newOpts.map(option => {
				optionMap.set(option.value.id, option);
			});

			await this.setState({
				options: newOpts,
				optionMap
			});

			return newOpts;
		}
		return null;
	}

	initValue = async () => {
		const value = this.props.value;

		// find option belong to value
		const parsedValue: Array<IOption> = [];
		value.map((item) => {
			let key = '';
			if (item) {
				if (item.key) {
					key = item.key;
				} else if (item.id) {
					key = item.id;
				} else if (item.value?.id) {
					key = item.value.id;
				} else {
					key = item;
				}
			}
			const val = this.state.optionMap.get(`${key}`);
			if (val) {
				parsedValue.push(this.state.optionMap.get(`${key}`));
			}
		});
		await this.setState({ value: parsedValue });

	}

	componentDidMount = async () => {
		await this.initValuePreProcessOptions(this.props.options);
		// await this.preProcessOptions(this.props.options);
		// await this.initValue();
		this.setState({ isInit: false });
	}

	componentDidUpdate = (prevProps: IProps, prevState: IState, snapshot) => {
		if (prevProps.options?.length != this.props.options?.length) {
			this.initValuePreProcessOptions(this.props.options);
			// this.preProcessOptions(this.props.options);
		}
	}

	createGroup = (groupName: string, groupMap: Map<string, GroupInfo>, basePadding = 0, level = 0) => {
		const groupInfo: GroupInfo = groupMap.get(groupName);
		let allProps = (groupInfo.current ? groupInfo.current.props : {});
		const { classes } = this.props;

		// if has children ==> it is the group object
		if (groupInfo && groupInfo.children.length > 0) {
			const zIndex = 1000 - level;
			allProps = {
				...allProps,
				className: `${allProps.className} ${level == 0 ? classes.mainGroupHeader : classes.subGroupHeader}`
			};

			const content = () => (
				<List className={classes.groupBox} data-level={level} key={Math.random()}
					subheader={(
						<ListSubheader {...allProps} style={level == 0 ? {} : { top: Number(35 * level), color: indigo[500], zIndex }}>
							<Grid container alignItems='center' wrap='nowrap' className={classes.mainGroupHeaderContent}>
								{level == 0 && (
									<Fragment>
										<MDIcon size={'13pt'} path={this.props.mainGroupIcon || mdiFlare} />&nbsp;
									</Fragment>
								)}
								<Box lineHeight='0'>{groupName}</Box>
							</Grid>
						</ListSubheader>
					)}>
					{
						groupInfo.children.map((name) => this.createGroup(name, groupMap, 30, level + 1))
					}
				</List>
			);

			return level == 0 ? content() : <ListItem key={groupName} style={{ padding: 0, paddingLeft: Number(15 * level), width: 'auto' }}>{content()}</ListItem>;
		}

		// if NOT have children ==> it is the empty main_group object (level = 0) OR group's item object (level > 0)

		return level == 0 ? (
			<List key={Math.random()} className={classes.groupBox}
				subheader={(
					<ListSubheader {...allProps}>
						<Grid container alignItems='center' wrap='nowrap' className={classes.mainGroupHeaderContent}>
							<MDIcon size={'13pt'} path={(this.props.mainGroupIcon || mdiFlare)} />&nbsp;<Box>{groupName}</Box>
						</Grid>
					</ListSubheader>
				)}>
			</List>
		) : (
			<ListItem key={Math.random()} {...allProps} style={{ marginLeft: Number(15 * level), width: 'auto' }}>
				<Grid container alignItems='center' wrap='nowrap' className={classes.mainGroupHeaderContent}>
					&nbsp;&nbsp;<Box fontSize={14}> {groupName}</Box>
				</Grid>
			</ListItem>
		);

	}

	public render = (): React.ReactNode => {
		const { open, ...otherProps } = this.props;
		delete otherProps['classes'];

		return this.state.isInit ? (<div></div>) : (
			<ReactAutosuggest {...otherProps}
				fullWidth
				componentName={this.props.componentName}
				freeSolo={false}

				// GraphQL remote API
				// remoteUrl={this.props.remoteUrl}
				// gqlMethod={this.props.gqlMethod}
				// queryString={this.props.queryString}
				// parseGqlData={this.props.parseGqlData}
				// convertOptions={this.preProcessOptions}

				{...(open ? { open } : {})}
				renderTags={(values, getTagProps) => {

					return values.map((option, index) => {
						if (option) {
							const parts: Array<string> = option.value.dir.split('|');

							return (
								<Chip key={`tag_${index}`} size='small' variant='default' style={{ borderColor: grey[500] }}
									onDelete={() => true}
									label={(
										<Fragment>
											{/* <span style={option.value.level > 0 ? { borderRight: `solid ${grey[400]} 1px`, paddingRight: 10, marginRight: 10 } : {}}>{option.value.groupName}</span> */}
											{
												option.value.level > 0 ? (
													<Tooltip title={
														<Fragment>
															{
																parts.map((part, index) => {
																	if (index == 0) {
																		return <Fragment key={index}><b>{option.value.groupName}</b><br /></Fragment>;
																	}

																	const spaces = [];
																	for (let i = 0; i < index; i++) {
																		spaces.push(<Fragment key={i}>&nbsp;&nbsp;</Fragment>);
																	}

																	return (
																		<Fragment key={index}>
																			{spaces}⎩ {part}<br />
																		</Fragment>
																	);
																})
															}
														</Fragment>
													}>
														<Box style={{ cursor: 'pointer' }}>{parts[parts.length - 1]}</Box>
													</Tooltip>
												) : (
													<Box>{option.value.dir}</Box>
												)
											}
										</Fragment>
									)}
									{...getTagProps({ index })}
								/>
							);
						}
						return false;
					});
				}}

				getOptionSelected={(optionItem, selectedItem) => {
					// this function is using for consider if the selectedItem is the same with optionItem
					// If true (the same together) => NOT show in the suggestion
					return optionItem.key.startsWith(selectedItem.key) || selectedItem.key.startsWith(optionItem.key);
				}}

				renderOption={(option) => {
					return option.value.dir;
				}}

				groupBy={(option) => {
					return option.value.groupName;
				}}

				renderGroup={(option: { group: string, children: Array<any> }) => {
					const { group: groupName, children: groupChildren } = option;
					const groupMap = new Map<string, GroupInfo>();

					// init
					groupMap.set(groupName, {
						name: groupName,
						children: []
					});

					groupChildren.forEach((item) => {
						const parts: Array<string> = item.props.children.split('|');

						const partsLength = parts.length;

						if (partsLength == 1 && parts[0] == groupName) {
							// this is the Main group => it mean user can select this one
							// add Autocomplete generated <li> 
							groupMap.get(groupName).current = item;

						} else {
							for (let i = 1; i < partsLength; i++) {
								const name = parts[i];
								if (!groupMap.get(name)) {
									const obj = {
										name,
										children: []
									};
									// ex: given item with `org` is "A|B|C" => we have 3 groups [A, B, C] in `groupMap`
									// but, only item C can be selected by user (Autocomplete has generated the <li> element for this item)
									// => only add the generated <li> into `current` to create the option on suggestion list
									if (i == partsLength - 1) {
										obj.current = item;
									}
									groupMap.set(name, obj);
								}

								const parent = groupMap.get(parts[i - 1]);
								if (parent && !parent.children.includes(name)) {
									parent.children.push(name);
								}
							}
						}
					});

					return this.createGroup(groupName, groupMap);
				}}

				onChange={this.props.onChange}
				options={this.state.options}
				value={this.state.value}
			/>
		);


	}
}

export default withStyles(useStyles)(MultiLvTagInput);



